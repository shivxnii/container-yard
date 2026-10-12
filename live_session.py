"""Live simulation: a video file (played in real time) or a camera drives the room map.

Three cooperating parts, so the dashboard video never freezes while the analysis is busy:
  reader thread  : decodes + shows the (annotated) video, registers frames to one board coordinate system
  OCR thread     : reads the handwritten tag codes (#LQ6872 ...) on the newest frame
  board model    : tags -> positions -> zone / segment / slot / tier; detects moved and removed tags
A video file is slowed down automatically if the OCR cannot keep up, so no tag is skipped.

Needs: pip install opencv-python rapidocr-onnxruntime
Env (optional): CAMERA_FIXED=1  camera does not move (skip registration)     LIVE_STEP_S=0.35  seconds between analysis steps
                OCR_WIDTH=1280  frame width given to the OCR (bigger = reads smaller text, slower)
                LIVE_HWACCEL=0  do not try hardware video decoding           LIVE_LAG_S=1.5   max seconds the OCR may fall behind
"""
import math, os, threading, time
from pathlib import Path
import numpy as np
import db
import tag_reader
import video_processor as vp
from board import Board, CONFIRM

cv2 = vp.cv2
STEP_S = float(os.getenv("LIVE_STEP_S", "0.35"))
OCR_WIDTH = int(os.getenv("OCR_WIDTH", "1280"))
LAG_S = float(os.getenv("LIVE_LAG_S", "1.5"))
FIXED = os.getenv("CAMERA_FIXED") == "1"
HWACCEL = os.getenv("LIVE_HWACCEL", "1") != "0"
SMALL_W = 640


def _open(src):
    if isinstance(src, int):
        if os.name == "nt":                                  # DirectShow is the reliable backend for webcams on Windows
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(src)
    if HWACCEL and Path(str(src)).is_file() and hasattr(cv2, "VIDEO_ACCELERATION_ANY"):
        try:                                                 # hardware decoding makes big 1080p/4K .MOV files much lighter
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
            ok, f = cap.read()
            if cap.isOpened() and ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap
            cap.release()
        except Exception:
            pass
    return cv2.VideoCapture(src)


def _plain(o):
    """numpy numbers -> plain Python (the JSON encoder of the API cannot handle numpy types)."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return [_plain(v) for v in o.tolist()]
    return o


def _register(prev, cur, bf):
    """Frame -> keyframe transform: a similarity first (robust), refined to a full homography when many matches
    agree - a whiteboard is a plane, so a handheld camera moves it by a homography and a similarity alone drifts."""
    k1, d1 = prev; k2, d2 = cur
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    m = sorted(bf.match(d2, d1), key=lambda m: m.distance)[:400]
    if len(m) < 20:
        return None
    src = np.float32([k2[x.queryIdx].pt for x in m]); dst = np.float32([k1[x.trainIdx].pt for x in m])
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is None or inl.sum() < 15:
        return None
    Hs = np.vstack([A, [0, 0, 1]])
    ok = inl.ravel().astype(bool)
    if ok.sum() >= 30:
        try:
            Hh, ih = cv2.findHomography(src[ok], dst[ok], cv2.RANSAC, 2.0)
        except cv2.error:
            Hh = None
        if (Hh is not None and abs(Hh[2, 2]) > 1e-9 and ih.sum() >= .8 * ok.sum()):
            Hh = Hh / Hh[2, 2]
            if abs(Hh[2, 0]) < 2e-3 and abs(Hh[2, 1]) < 2e-3 and .5 < np.linalg.det(Hh[:2, :2]) < 2:
                return Hh
    return Hs


class Registrar:
    """Maps every frame (640 px gray) to the board coordinate system = the first frame. Keyframes avoid drift."""

    def __init__(self):
        self.orb = cv2.ORB_create(1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kfs = []                                       # (keypoints+descriptors, H keyframe->board)

    def register(self, gray, keyframe=False):
        feats = self.orb.detectAndCompute(gray, None)
        self.last_feats = feats
        if not self.kfs:
            if feats[1] is None or len(feats[0]) < 80:
                return None
            self.kfs.append((feats, np.eye(3)))
            return np.eye(3)
        H, W = gray.shape
        cands = self.kfs[-4:] + ([self.kfs[0]] if len(self.kfs) > 4 else [])
        for kf, Hk in reversed(cands):                       # newest keyframe first: relative registration is the most accurate
            Hm = _register(kf, feats, self.bf)                       # this frame -> keyframe
            if Hm is not None and .1 < abs(np.linalg.det(Hm[:2, :2])) < 10:
                Hc = Hk @ Hm
                c = Hm @ np.array([W / 2, H / 2, 1.0])
                if keyframe or math.hypot(c[0] - W / 2, c[1] - H / 2) > .25 * W or not .8 < math.sqrt(abs(np.linalg.det(Hm[:2, :2]))) < 1.25:
                    self.kfs.append((feats, Hc))                     # moved a lot: new keyframe
                    self.kfs = self.kfs[:1] + self.kfs[-59:]
                return Hc
        return None


class LiveSession:
    def __init__(self, source, layout, job_no):
        self.source, self.layout, self.job_no = source, layout, job_no
        self.is_file = isinstance(source, str) and Path(source).is_file()
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.jpg = None
        self.vocab = tag_reader.vocab_from_layout(layout, Path(__file__).with_name("tags.txt"))
        self.board = Board(layout, self.vocab)
        self.state = {"status": "starting", "message": "Opening source…", "kind": "file" if self.is_file else "camera",
                      "source": str(source), "session": job_no, "progress": 0, "frame": 0, "calibrated": False,
                      "in_view": 0, "containers": [], "view": None, "position": None,
                      "stats": {"ocr_passes": 0, "tags_read": 0, "lost": 0}}
        self.reader_thread = threading.Thread(target=self._run, daemon=True)
        self.ocr_thread = threading.Thread(target=self._ocr_loop, daemon=True)
        self.job = None                                     # newest frame waiting for the OCR
        self.job_cv = threading.Condition()
        self.ocr_busy, self.ocr_vt = False, 0.0            # OCR state / video time of the last frame it finished
        self.H_step = None                                  # latest small-frame -> board transform (+ gray used)
        self.step_small = None
        self.overlay = []
        self.clean, self.boxes = None, {}                    # newest un-annotated frame + tag boxes in it (for the detected-container picture)
        self.written, self.hud = {}, ""
        self.reg = None
        self.anchor, self.rel, self._shift = None, None, (0.0, 0.0)   # newest tag reads in their own picture + how the current picture relates to it
        self._rec = []

    # ----- control -----
    def start(self):
        self.reader_thread.start()
        self.ocr_thread.start()

    def stop(self):
        self.stop_evt.set()
        with self.job_cv:
            self.job_cv.notify_all()
        self.reader_thread.join(3)

    def alive(self):
        return self.reader_thread.is_alive() and not self.stop_evt.is_set()

    @property
    def thread(self):                                       # compatibility
        return self.reader_thread

    def snapshot(self):
        with self.lock:
            s = dict(self.state)
            s["stats"] = dict(s["stats"])
        return _plain(s)

    def _set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def _bump(self, **kw):
        with self.lock:
            for k, v in kw.items():
                self.state["stats"][k] += v

    def _summary(self):
        with self.lock:
            tags = [c["container_id"] for c in self.state["containers"]]
            st = self.state["stats"]
        if not tags:
            return (f"Video finished · no tags were read ({st['ocr_passes']} OCR passes). "
                    "Is the board in view and in focus?")
        return f"Video finished · {len(tags)} tag(s) on the map: " + ", ".join(tags)

    # ----- reader thread: decode, show, register, hand frames to the OCR -----
    def _run(self):
        try:
            self._loop()
        except Exception as e:
            self._set(status="error", message=str(e))
        finally:
            with self.job_cv:
                self.job_cv.notify_all()

    def _loop(self):
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed:  pip install opencv-python")
        cap = _open(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {self.source!r}. For a webcam try another index (0, 1, 2); "
                               "for a file try an H.264 .mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        fps = fps if 1 <= fps <= 130 else 30.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) if self.is_file else 0
        step = max(1, int(fps * STEP_S))
        show_every = 3 if fps > 40 else (2 if fps > 28 else 1) if self.is_file else 1
        self.reg = Registrar()
        self._set(status="running", message="Loading the tag reader…")
        idx, fails, t0, last_step = 0, 0, time.time(), -10 ** 9
        try:
            while not self.stop_evt.is_set():
                vt = idx / fps
                if self.is_file:
                    # never run ahead of the tag reader: wait (the picture holds) until it has caught up
                    if vt - self.ocr_vt > LAG_S and (self.ocr_busy or self.job is not None):
                        t0 += 0.05                          # keep the playback clock from jumping ahead afterwards
                        time.sleep(0.05); continue
                    lag = time.time() - (t0 + vt)
                    if lag < 0:
                        time.sleep(min(0.04, -lag)); continue
                    if lag > 0.5:
                        t0 += lag                           # decoding is slower than the video: just play slower
                if cap.grab():
                    idx += 1
                    due = idx - last_step >= step
                    if self.is_file and not due and idx % show_every:
                        continue                            # decode only the pictures we show or analyse
                    ok, frame = cap.retrieve()
                else:
                    ok = False
                if not ok:
                    if self.is_file:
                        self._finish_file()
                        return
                    fails += 1
                    if fails > 30:
                        raise RuntimeError("The camera stopped returning frames")
                    time.sleep(0.1); continue
                fails = 0
                k = SMALL_W / frame.shape[1]
                small = cv2.resize(frame, None, fx=k, fy=k)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if due:
                    last_step = idx
                    self._analyse(frame, gray, idx / fps)
                self._publish_frame(small, gray, idx, total)
        finally:
            cap.release()
            if self.stop_evt.is_set():
                self._set(status="stopped", message="Stopped")

    def _finish_file(self):
        t = time.time()
        while (self.ocr_busy or self.job is not None) and time.time() - t < 20 and not self.stop_evt.is_set():
            time.sleep(0.1)                                 # let the last frame be read
        self._set(status="ended", progress=1, message=self._summary())

    def _analyse(self, frame, gray, vt):
        if FIXED:
            H = np.eye(3)
        else:
            H = self.reg.register(gray, keyframe=self.job is None and not self.ocr_busy)   # the frame handed to the tag reader becomes a keyframe,
            # so the picture drawn later is registered directly against it (errors cancel: boxes stay on the text)
            if H is None:
                if self.reg.kfs:
                    self._bump(lost=1)
                    self._set(message="Lost track of the board (camera moved too far or the picture is blurred)")
                self.H_step = None
                return
        self.H_step, self.step_small = H, cv2.resize(gray, None, fx=.25, fy=.25)
        self.rel = None
        a = self.anchor
        if a is not None and not FIXED:                      # register this picture straight against the one the tags were read in
            Ma = _register(a["feats"], self.reg.last_feats, self.reg.bf)
            if Ma is not None and .3 < abs(np.linalg.det(Ma[:2, :2])) < 3:
                self.rel = (Ma, a)
        if self.job is None and not self.ocr_busy:
            k = OCR_WIDTH / frame.shape[1]
            big = cv2.resize(frame, None, fx=k, fy=k) if k < 1 else frame
            kk = SMALL_W / big.shape[1]                      # OCR-frame px -> small-frame px
            with self.job_cv:
                self.job = (big, H @ np.diag([kk, kk, 1.0]), vt, None if FIXED else self.reg.last_feats, kk)
                self.job_cv.notify()

    # ----- OCR thread -----
    def _ocr_loop(self):
        try:
            eng, err = tag_reader.load_engine()
            if eng is None:
                self._set(status="error", message=err)
                self.stop_evt.set()
                return
            self._set(message="Looking for the board…")
            while not self.stop_evt.is_set():
                with self.job_cv:
                    while self.job is None and not self.stop_evt.is_set() and (self.reader_thread.is_alive() or self.job):
                        self.job_cv.wait(0.2)
                    job, self.job = self.job, None
                    self.ocr_busy = job is not None
                if job is None:
                    if not self.reader_thread.is_alive():
                        return
                    continue
                self._ocr_job(job)
        except Exception as e:
            self._set(status="error", message=f"OCR failed: {e}")

    def _ocr_job(self, job):
        frame, H, vt, feats, kk = job
        try:
            items, wh = tag_reader.read_frame(frame, OCR_WIDTH)
            reads, seen = tag_reader.parse(items, self.vocab)
            if os.getenv("LIVE_RECORD"):
                self._rec.append((vt, reads, seen, H.copy(), wh))
            self.board.update(reads, seen, H, wh)
            if feats is not None:
                self.anchor = {"feats": feats, "boxes": {r["id"]: (r["cx"] * kk, r["cy"] * kk, r["w"] * kk, r["h"] * kk) for r in reads}}
            self._publish_board(len(reads))
        finally:
            self.ocr_vt = vt
            self.ocr_busy = False

    def _publish_board(self, n_reads):
        rows = self.board.containers()
        self.board.take_events()
        for r in rows:                                      # history in the DB: new tags and every move
            key = (r["checkpoint"], r["x"], r["y"], r["z_tier"])
            if self.written.get(r["container_id"]) != key:
                self.written[r["container_id"]] = key
                db.add_sighting(container_id=r["container_id"], checkpoint=r["checkpoint"], x=r["x"], y=r["y"],
                                z_tier=r["z_tier"], heading_deg=90.0, heading_ambiguous=True, ocr_conf=r["ocr_conf"])
        live_ids = {r["container_id"] for r in rows}
        for cid in list(self.written):
            if cid not in live_ids:
                del self.written[cid]                        # removed: if it comes back it is a new sighting
        self.overlay = self.board.overlay()
        with self.lock:
            self.state["containers"] = rows
            self.state["in_view"] = sum(r["in_view"] for r in rows)
            self.state["calibrated"] = self.board.model is not None
            self.state["view"] = self.board.view_polygon()
            self.state["stats"]["ocr_passes"] += 1
            self.state["stats"]["tags_read"] += n_reads
            self.state["message"] = f"{len(rows)} tag(s) on the map · {n_reads} read in the last pass"

    def crop_jpeg(self, tid):
        """JPEG of the area around one tag in the newest frame (None when it is not in the picture)."""
        clean, b = self.clean, self.boxes.get(str(tid).upper())
        if clean is None or b is None:
            return None
        x0, y0, x1, y1 = b; w, h = x1 - x0, y1 - y0
        H, W = clean.shape[:2]
        a, c = max(0, int(x0 - .7 * w)), max(0, int(y0 - 2.4 * h))
        e, f = min(W, int(x1 + .7 * w)), min(H, int(y1 + 2.4 * h))
        if e - a < 10 or f - c < 10:
            return None
        img = clean[c:f, a:e].copy()
        cv2.rectangle(img, (int(x0 - a), int(y0 - c)), (int(x1 - a), int(y1 - c)), (80, 200, 80), 2)
        k = 480 / img.shape[1]
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpg.tobytes() if ok else None

    # ----- annotated video (drawn on every shown frame, so it stays smooth) -----
    def _current_H(self, gray):
        """Transform current small frame -> raw registered board coords: the last analysed frame's transform,
        moved by the cheap shift estimate between that frame and this one."""
        Hs = self.H_step
        if Hs is None or self.step_small is None:
            return None
        g4 = cv2.resize(gray, None, fx=.25, fy=.25)
        try:
            (dx, dy), resp = cv2.phaseCorrelate(np.float32(self.step_small), np.float32(g4))
        except cv2.error:
            dx = dy = resp = 0
        if resp < .05 or abs(dx) > g4.shape[1] * .5 or abs(dy) > g4.shape[0] * .5:
            dx = dy = 0
        self._shift = (dx * 4, dy * 4)                       # current picture = analysed picture moved by this
        return Hs @ np.array([[1, 0, -dx * 4], [0, 1, -dy * 4], [0, 0, 1.0]])

    def _project_overlay(self, shape, Hn):
        """Tags -> boxes in small-frame pixels: [(id, x0, y0, x1, y1, confirmed)]."""
        res = {}
        try:
            Hi = np.linalg.inv(Hn)
        except np.linalg.LinAlgError:
            return []
        sc = math.sqrt(abs(np.linalg.det(Hn[:2, :2])))
        for (tid, pos, w, h, conf, inview) in self.overlay:
            p = Hi @ np.array([pos[0], pos[1], 1.0]); p = p[:2] / p[2]
            hw, hh = w / sc / 2 * 1.15, h / sc / 2 * 1.35
            if hw < 5 or not (0 < p[0] < shape[1] and 0 < p[1] < shape[0]):
                continue
            res[tid] = (tid, p[0] - hw, p[1] - hh, p[0] + hw, p[1] + hh, conf)
        rel = self.rel
        if rel is not None:                                  # tags of the newest read: carried over from their own picture
            Ma, a = rel
            try:
                Mi = np.linalg.inv(Ma)
            except np.linalg.LinAlgError:
                Mi = None
            if Mi is not None:
                s2 = math.sqrt(abs(np.linalg.det(Mi[:2, :2])))
                for tid, (cx, cy, w, h) in a["boxes"].items():
                    p = Mi @ np.array([cx, cy, 1.0]); p = p[:2] / p[2]
                    px, py = p[0] + self._shift[0], p[1] + self._shift[1]
                    hw, hh = w * s2 / 2 * 1.15, h * s2 / 2 * 1.35
                    if hw < 5 or not (0 < px < shape[1] and 0 < py < shape[0]):
                        continue
                    old = res.get(tid)
                    res[tid] = (tid, px - hw, py - hh, px + hw, py + hh, old[5] if old else True)
        return list(res.values())

    def _publish_frame(self, small, gray, idx, total):
        vis = small
        clean, boxes = small.copy(), {}
        Hn = self._current_H(gray)
        if Hn is not None:
            for (tid, x0, y0, x1, y1, conf) in self._project_overlay(vis.shape, Hn):
                boxes[tid] = (x0, y0, x1, y1)
                col = (80, 200, 80) if conf else (0, 165, 255)
                cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
                cv2.putText(vis, tid, (int(x0), max(14, int(y0) - 5)), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 2, cv2.LINE_AA)
            # where the camera is looking, on the layout map: the middle of the picture (one moving dot)
            c = Hn @ np.array([vis.shape[1] / 2, vis.shape[0] / 2, 1.0])
            pos = self.board.layout_point(c[:2] / c[2])
            if pos is not None:
                self._set(position=pos)
        self.clean, self.boxes = clean, boxes
        with self.lock:
            n = len(self.state["containers"]); msg = "tracking" if self.H_step is not None else "searching…"
        cv2.putText(vis, f"tags on map: {n}  ·  {msg}", (8, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"tags on map: {n}  ·  {msg}", (8, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 30), 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            self.jpg = jpg.tobytes()
        self._set(frame=idx, progress=round(min(1, idx / total), 3) if total else 0)


# ----- module-level single session -----
_cur, _count = None, 0


def start(source, layout):
    global _cur, _count
    stop()
    _count += 1
    _cur = LiveSession(source, layout, _count)
    _cur.start()
    return _cur


def stop():
    if _cur is not None and _cur.alive():
        _cur.stop()


def reset():
    """Stop the session and forget it: the dashboard goes back to the (empty) database view."""
    global _cur
    stop()
    _cur = None


def state():
    return _cur.snapshot() if _cur else {"status": "idle", "message": "", "containers": [], "in_view": 0, "view": None}


def crop(tid):
    return _cur.crop_jpeg(tid) if _cur else None


def mjpeg():
    s, last = _cur, None
    while s is not None and s is _cur:
        if s.jpg is not None and s.jpg is not last:
            last = s.jpg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last + b"\r\n"
        if not s.alive():
            break
        time.sleep(0.03)