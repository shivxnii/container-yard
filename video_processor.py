"""Uploaded video -> zone / segment / tier sightings (baseline, classical OpenCV).

Pipeline: sample frames -> find drawn container boxes -> register frames to one reference
(ORB + homography, so a handheld pan still works) -> merge repeated sightings -> place on the
room layout (zone / segment / slot) -> rough z (stack tier).
Swap `find_boxes` for your own detector (YOLO-OBB etc.) if you have one: it only has to return
[(cx, cy, w, h), ...] in frame pixels.  Needs: pip install opencv-python
"""
import math, re, os
from collections import Counter
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
pytesseract = None
if os.getenv("OCR") == "1":          # optional: OCR=1 + pip install pytesseract (+ Tesseract app) reads the handwritten tag
    try:
        import pytesseract
    except ImportError:
        pass

CONTAINER_H_M = 2.59
Z_GAP = float(os.getenv("Z_GAP", "0.15"))       # boxes closer than this x box-height in a column = stacked
MAX_TIER = 5
Z_MODE = os.getenv("Z_MODE", "stack")            # "stack" | "flat"
WIDTH = 640
STEP_S = 0.4
TAG = re.compile(r"[A-Z0-9]{5,8}")


def find_boxes(gray):
    """Drawn container boxes in one frame -> [(cx, cy, w, h)], w >= h.
    A box is outline + hatched door end, so its white interior is split into pieces; pieces on the same row are merged."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 10)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cn, hi = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    H, W = th.shape
    pcs = []                                              # (x0, y0, x1, y1) of inner white regions
    for c, h in zip(cn, hi[0]):
        if h[3] == -1:
            continue
        a = cv2.contourArea(c)
        x, y, w, hh = cv2.boundingRect(c)
        if a < .002 * H * W or a > .25 * H * W or x < 4 or y < 4 or x + w > W - 4 or y + hh > H - 4:
            continue
        if a / (w * hh) < .7:
            continue
        pcs.append([x, y, x + w, y + hh])
    pcs.sort()
    merged = True
    while merged:                                         # union pieces that sit side by side on one row
        merged = False
        for i in range(len(pcs)):
            for j in range(i + 1, len(pcs)):
                p, q = pcs[i], pcs[j]
                ov = min(p[3], q[3]) - max(p[1], q[1])
                hmin = min(p[3] - p[1], q[3] - q[1])
                gap = max(q[0] - p[2], p[0] - q[2])
                if ov > .7 * hmin and gap < .5 * hmin:
                    pcs[i] = [min(p[0], q[0]), min(p[1], q[1]), max(p[2], q[2]), max(p[3], q[3])]
                    del pcs[j]; merged = True; break
            if merged:
                break
    out = []
    for x0, y0, x1, y1 in pcs:
        w, h = x1 - x0, y1 - y0
        if h > 0 and 2.4 <= w / h <= 9:
            out.append(((x0 + x1) / 2, (y0 + y1) / 2, float(w), float(h)))
    return out


def _register(prev, cur, orb, bf):
    k1, d1 = prev; k2, d2 = cur
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    m = sorted(bf.match(d2, d1), key=lambda m: m.distance)[:300]
    if len(m) < 20:
        return None
    src = np.float32([k2[x.queryIdx].pt for x in m]); dst = np.float32([k1[x.trainIdx].pt for x in m])
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)   # pan + zoom + roll
    if A is None or inl.sum() < 15:
        return None
    return np.vstack([A, [0, 0, 1]])


def _ocr(gray, cx, cy, w, h):
    if pytesseract is None:
        return None
    x0, x1 = int(cx - w / 2), int(cx + w / 2); y0, y1 = int(cy - h / 2), int(cy + h / 2)
    crop = gray[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, None, fx=2, fy=2)
    try:
        t = pytesseract.image_to_string(crop, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#")
    except Exception:
        return None
    m = TAG.search(re.sub(r"[^A-Z0-9]", "", t.upper()))
    return m.group(0) if m else None


def process(path, layout, job_no, progress=lambda p, msg="": None, preview_path=None):
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed:  pip install opencv-python")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video (try converting it to H.264 .mp4)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
    step = max(1, int(fps * STEP_S))
    orb = cv2.ORB_create(1500); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    tracks = []            # in reference coords: dict(c=[x,y], w, h, n, tags=Counter)
    H_ref = None; prev = None; best = (0, None); idx = 0
    while True:
        idx += 1
        if (idx - 1) % step:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        progress(min(0.9, idx / total), "Detecting boxes")
        s = WIDTH / frame.shape[1]
        small = cv2.resize(frame, None, fx=s, fy=s)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        cur = (kp, des)
        boxes = find_boxes(gray)
        if H_ref is None:
            if not boxes:
                prev = cur; continue                     # wait until the board is in view
            H_ref = np.eye(3)
        else:
            Hm = _register(prev, cur, orb, bf)
            if Hm is None:
                prev = cur; continue
            H_ref = H_ref @ Hm
            if not (.02 < abs(np.linalg.det(H_ref[:2, :2])) < 50):
                break                                    # registration drifted
        prev = cur
        for (cx, cy, w, h) in boxes:
            p = H_ref @ np.array([cx, cy, 1.0]); p = p[:2] / p[2]
            sc = math.sqrt(abs(np.linalg.det(H_ref[:2, :2])))
            w, h = w * sc, h * sc
            t = next((t for t in tracks if abs(t["c"][0] - p[0]) < .8 * t["w"] and abs(t["c"][1] - p[1]) < .6 * t["h"]), None)
            if t is None:
                t = {"c": p, "w": w, "h": h, "n": 0, "tags": Counter()}; tracks.append(t)
            t["n"] += 1; t["c"] = (t["c"] * (t["n"] - 1) + p) / t["n"]
            t["w"] += (w - t["w"]) / t["n"]; t["h"] += (h - t["h"]) / t["n"]
            tag = _ocr(gray, cx, cy, w / sc, h / sc)
            if tag:
                t["tags"][tag] += 1
        if preview_path and len(boxes) > best[0]:
            vis = small.copy()
            for (cx, cy, w, h) in boxes:
                cv2.rectangle(vis, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (0, 0, 255), 2)
            best = (len(boxes), vis)
    cap.release()
    if preview_path and best[1] is not None:
        cv2.imwrite(preview_path, best[1])
    progress(0.95, "Mapping to zones")
    tracks = [t for t in tracks if t["n"] >= 4]
    return place(tracks, layout, job_no)


def place(tracks, layout, job_no, calib=None):
    """Reference-frame boxes -> segment (by column), slot row (by height), zone from the segment, rough tier.
    calib=None : one-shot behaviour (video upload).
    calib={}   : same, and the column / row calibration used is stored in it.
    calib filled: reused as-is (live mode) so one moved box cannot shift every other box."""
    if not tracks:
        return []
    frozen = bool(calib) and "colx" in calib
    mw = calib["mw"] if frozen else float(np.median([t["w"] for t in tracks]))
    tracks = [t for t in tracks if t["w"] > .6 * mw]                      # drop slivers / partial boxes
    if not tracks:
        return []
    segs = [(z["name"], g) for z in layout["zones"] for g in z["segments"]]
    if frozen:
        cols = [[] for _ in calib["colx"]]
        for t in tracks:
            cols[int(np.argmin([abs(t["c"][0] - x) for x in calib["colx"]]))].append(t)
        seg_of, lo, hi = calib["seg_of"], calib["lo"], calib["hi"]
    else:
        cols = []                                                          # columns = boxes sharing an x position
        for t in sorted(tracks, key=lambda t: t["c"][0]):
            if cols and t["c"][0] - np.mean([u["c"][0] for u in cols[-1]]) < .6 * mw:
                cols[-1].append(t)
            else:
                cols.append([t])
        while len(cols) > len(segs):                                       # too many columns: merge the closest pair
            d = [abs(np.mean([u["c"][0] for u in cols[i + 1]]) - np.mean([u["c"][0] for u in cols[i]])) for i in range(len(cols) - 1)]
            i = int(np.argmin(d)); cols[i] += cols.pop(i + 1)
        n = len(cols)
        seg_of = [round(i * (len(segs) - 1) / max(n - 1, 1)) if n < len(segs) else i for i in range(n)]
        ys = np.array([t["c"][1] for t in tracks]); lo, hi = np.percentile(ys, [3, 97])
        if calib is not None:
            calib.update(mw=mw, colx=[float(np.mean([u["c"][0] for u in c])) for c in cols],
                         seg_of=[int(k) for k in seg_of], lo=float(lo), hi=float(hi))
    rows = []; count = 0
    for ci, col in enumerate(cols):
        zone, g = segs[seg_of[ci]]
        free = list(g["slots"])
        ymax, ymin = max(sl["y"] for sl in g["slots"]), min(sl["y"] for sl in g["slots"])
        # rough z: boxes in one column that (almost) touch are one stack; lowest on the board = tier 1
        stacks, cur = [], []
        for t in sorted(col, key=lambda u: -u["c"][1]):
            if Z_MODE == "stack" and cur and (cur[-1]["c"][1] - t["c"][1]) - t["h"] < Z_GAP * t["h"] and len(cur) < MAX_TIER:
                cur.append(t)
            else:
                if cur:
                    stacks.append(cur)
                cur = [t]
        if cur:
            stacks.append(cur)
        for st in sorted(stacks, key=lambda st: np.mean([u["c"][1] for u in st])):
            ty = ymax - (st[0]["c"][1] - lo) / max(hi - lo, 1e-6) * (ymax - ymin)
            if not free:
                break
            sl = min(free, key=lambda s_: abs(s_["y"] - ty)); free.remove(sl)
            for k, t in enumerate(st, 1):                      # whole stack shares one footprint
                count += 1
                tag = t["tags"].most_common(1)[0][0] if t["tags"] else (t.get("id") or f"V{job_no}-B{count:02d}")
                rows.append({"container_id": tag[:11], "zone": zone, "segment": g["name"], "x": round(float(sl["x"]), 2),
                             "y": round(float(sl["y"]), 2), "z_tier": k, "z_m": round(k * CONTAINER_H_M, 2),
                             "votes": int(t["n"]), "ocr": bool(t["tags"]), "in_view": bool(t.get("in_view"))})
    return rows