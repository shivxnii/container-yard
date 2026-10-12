"""Board model for the live simulation.

Every confirmed tag read from the video has a position in one fixed "board" coordinate system (frames are
registered to it, so a panning camera is fine). From those positions the tag is placed on layout.json:
    column  -> segment, row -> slot, same (segment, slot) twice -> stacked (tier).
The column / row grid is fitted on the tags that are known from layout.json (robust median fit), so it is
right even when only part of the board is in view. Without known tags it falls back to clustering.
A tag that is read at a new place (twice) has moved; a tag that is not found where the camera looks for it is removed.
"""
import math, os
import numpy as np
import cv2

CONFIRM = 2          # reads before a tag appears on the map
MISS_MAX = 6         # OCR passes a tag may be missing (while its place is in view and big enough) before it is removed
SETTLED = 6         # reads after which a tag counts as settled (moving it raises an event)
MOVE_CONFIRM = 3     # consistent reads at a new place before the tag counts as moved
RAW_AGE = 4          # OCR passes a tag's last measured position is used for drawing on the video, then the (drift-corrected) model
MIN_TEXT_PX = 28     # text narrower than this (in the OCR frame) is too small to expect a read
CONTAINER_H_M = 2.59
STACK_COLUMNS = os.getenv("STACK_COLUMNS") == "1"   # show each whiteboard column as one stack (z) instead of one box per slot
PITCH_PER_TEXT_H = 4.2   # fallback: row pitch / text height on the whiteboard
SPACING_PER_TEXT_W = 7.2  # fallback: column spacing / text width


class Tag:
    def __init__(self, tid, pos, w, h, known, step):
        self.id, self.pos, self.w, self.h, self.known = tid, np.array(pos, float), w, h, known
        self.n, self.miss, self.first, self.last = 0, 0, step, step
        self.conf, self.in_view, self.cand = 0.0, True, None
        self.votes = {tid: 1}
        self.si = self.ri = None            # last placement (hysteresis)
        self.raw = None                     # last read as measured: (raw board pos, text w, text h, step) - used to draw the box exactly on the text


class Board:
    def __init__(self, layout, vocab):
        self.layout, self.vocab = layout, vocab
        self.segs = [(z["name"], g) for z in layout["zones"] for g in z["segments"]]
        self.rows_y = [sorted((s["y"] for s in g["slots"]), reverse=True) for _, g in self.segs]   # top first
        self.cols_x = [float(np.mean([s["x"] for s in g["slots"]])) for _, g in self.segs]
        self.tags, self.step, self.model, self.foot = {}, 0, None, None
        self.land_step, self.travel, self._c = 0, 0.0, None
        self.corr = (np.zeros(2), np.zeros(2), 1.0)   # landmark correction: (frame pivot, board pivot, scale)
        self._last_rows = []
        self.events = []                    # ("moved"|"removed"|"added", id) since the last call to take_events()

    # ---------- update from one OCR pass ----------
    def update(self, reads, seen, H, wh):
        """reads/seen from tag_reader.parse (OCR-frame px); H maps OCR-frame px -> board coords; wh = frame size."""
        self.step += 1
        W, Hh = wh
        pt = lambda x, y: (lambda p: p[:2] / p[2])(H @ np.array([x, y, 1.0]))
        c0 = pt(W / 2, Hh / 2)
        sc0 = (np.linalg.norm(pt(W / 2 + 10, Hh / 2) - c0) + np.linalg.norm(pt(W / 2, Hh / 2 + 10) - c0)) / 20   # board units per px
        meas = []                                             # (read, position, text width, text height) straight from the tracking
        for r in reads:
            meas.append((r, pt(r["cx"], r["cy"]),
                         float(np.linalg.norm(pt(r["cx"] + r["w"] / 2, r["cy"]) - pt(r["cx"] - r["w"] / 2, r["cy"]))),
                         float(np.linalg.norm(pt(r["cx"], r["cy"] + r["h"] / 2) - pt(r["cx"], r["cy"] - r["h"] / 2)))))
        if self._c is not None:
            self.travel += float(np.linalg.norm(c0 - self._c))      # how far the camera moved since the last landmark
        self._c = c0
        self._landmark_correction(meas)                       # tags already on the board pin this frame down (no drift)
        corr = self.corr
        fix = lambda p: corr[1] + corr[2] * (np.asarray(p) - corr[0])
        m = .04
        foot = cv2.perspectiveTransform(np.float32([[[m * W, m * Hh], [(1 - m) * W, m * Hh], [(1 - m) * W, (1 - m) * Hh],
                                                     [m * W, (1 - m) * Hh]]]), H).reshape(-1, 2)
        self.foot = np.float32([fix(p) for p in foot]).reshape(-1, 1, 2)
        sc = sc0 * corr[2]
        got = set()
        for r, p, w, h in meas:
            p0, w0, h0 = p, w, h                                  # as measured in this frame (for the video overlay)
            p, w, h = fix(p), w * corr[2], h * corr[2]
            t = self.tags.get(r["id"])
            if t is None and not r["known"]:                    # unknown text: attach to a tag already at that place
                t = self._near(p, w, h)
                if t is not None:
                    if t.known:
                        got.add(t.id); t.last, t.miss = self.step, 0
                        continue
                    t.votes[r["id"]] = t.votes.get(r["id"], 0) + 1
                    best = max(t.votes, key=t.votes.get)
                    if best != t.id and best not in self.tags:
                        del self.tags[t.id]; t.id = best; self.tags[best] = t
            if t is None:
                t = Tag(r["id"], p, w, h, r["known"], self.step)
                self.tags[t.id] = t
                if r["known"]:                                   # a known tag explains any unknown guess at its place
                    for o in [o for o in self.tags.values() if not o.known and self._close(o, p, w, h)]:
                        del self.tags[o.id]
            else:
                if self._close(t, p, w, h):
                    t.cand = None
                    a = max(1.0 / (t.n + 1), .3) if t.n < 4 else .15      # settled tags barely move
                    t.pos = t.pos * (1 - a) + p * a
                    t.w += (w - t.w) * a; t.h += (h - t.h) * a
                else:                                            # read somewhere else: moved?
                    if t.cand is not None and self._close_pt(t.cand[0], p, w, h):
                        t.cand = (t.cand[0] * .5 + p * .5, t.cand[1] + 1)
                    else:
                        t.cand = (p, 1)
                    young = t.n < SETTLED
                    if t.cand[1] >= (2 if young else MOVE_CONFIRM):
                        t.pos, t.cand, t.si, t.ri = t.cand[0], None, None, None
                        t.w, t.h = w, h
                        if not young:
                            self.events.append(("moved", t.id))
            t.raw = (np.array(p0, float), w0, h0, self.step)
            t.n += 1; t.last, t.miss, t.in_view = self.step, 0, True
            t.conf = max(t.conf * .8, r["conf"]) if t.n > 1 else r["conf"]
            got.add(t.id)
            if t.n == CONFIRM:
                self.events.append(("added", t.id))
        to_ref = lambda x, y: fix(pt(x, y))
        # tags whose place the camera is looking at, but that were not read
        seen_ref = [to_ref(x, y) for x, y in seen]
        lm = [t.pos for t in self.tags.values() if t.id in got and t.n >= 3]      # tags read in this very frame
        for t in list(self.tags.values()):
            if t.id in got:
                continue
            edge = cv2.pointPolygonTest(self.foot, (float(t.pos[0]), float(t.pos[1])), True)   # distance to the picture edge
            t.in_view = edge >= 0
            if edge < 3 * t.h or t.w / sc < MIN_TEXT_PX:
                continue                                         # near / outside the edge, or too small to read: no evidence
            if not any(np.linalg.norm(t.pos - q) <= 5 * t.h for q in lm):
                continue                                         # no tag read right next to it: the position is not trustworthy
            if any(abs(sx - t.pos[0]) < t.w * 1.0 and abs(sy - t.pos[1]) < t.h * 1.8 for sx, sy in seen_ref):
                t.miss = 0                                       # something tag-like is still written there
                continue
            t.miss += 1
            if t.miss >= (MISS_MAX if t.n >= CONFIRM else 2):
                del self.tags[t.id]
                if t.n >= CONFIRM:
                    self.events.append(("removed", t.id))

    def _landmark_correction(self, meas):
        """Tags that are already on the board (>= 3 reads, not moving) tell how far the tracking has drifted:
        shift (and rescale by the text size) this frame so they land where they were established.
        Landmarks that disagree with the majority are tags that moved, not drift: they are outvoted."""
        land = [(self.tags[r["id"]], p, h) for r, p, w, h in meas
                if r["id"] in self.tags and self.tags[r["id"]].n >= 3]
        if not land:
            return                                            # no landmark in view: keep the correction of the last frame
        prior = lambda q: self.corr[1] + self.corr[2] * (q - self.corr[0])
        err = [t.pos - prior(p) for t, p, h in land]           # how far each landmark is from where the old correction puts it
        best = None
        for i, (t, p, h) in enumerate(land):                  # largest group of landmarks that agree; ties: the smallest change
            grp = [j for j in range(len(land)) if np.linalg.norm(err[j] - err[i]) < 3.0 * max(t.h, land[j][0].h)]
            key = (len(grp), -np.linalg.norm(np.mean([err[j] for j in grp], axis=0)))
            if best is None or key > best[0]:
                best = (key, grp)
        grp = [land[j] for j in best[1]]
        s = float(np.clip(np.median([t.h / max(h, 1e-6) for t, p, h in grp]), .6, 1.7))
        if len(grp) == 1:
            s = 1.0 + (s - 1.0) * .5                          # one tag: text size is only a weak hint
        qc = np.mean([p for t, p, h in grp], axis=0); pc = np.mean([t.pos for t, p, h in grp], axis=0)
        h0 = max(grp[0][0].h, 1e-6)
        change = np.linalg.norm(np.mean([err[land.index(g)] for g in grp], axis=0)) / h0   # in text heights
        limit = (2.5 + 0.12 * self.travel / h0) * (3.0 if len(grp) >= 2 else 1.0)   # drift grows with the distance travelled
        if change > limit:
            return                                            # too sudden to be drift: that tag moved, keep the old correction
        self.corr, self.land_step, self.travel = (qc, pc, s), self.step, 0.0

    def _close_pt(self, a, p, w, h):
        return abs(a[0] - p[0]) < 1.5 * w and abs(a[1] - p[1]) < 2.0 * h

    def _close(self, t, p, w, h):
        return abs(t.pos[0] - p[0]) < 1.5 * max(t.w, w) and abs(t.pos[1] - p[1]) < 2.0 * max(t.h, h)

    def _near(self, p, w, h):
        for t in self.tags.values():
            if self._close(t, p, w, h):
                return t
        return None

    def fix(self, p):
        """raw registered board coords -> corrected board coords (the landmark correction)."""
        c = self.corr
        return c[1] + c[2] * (np.asarray(p, float) - c[0])

    def unfix(self, q):
        """corrected board coords -> raw registered coords (what a frame's transform H maps pixels to)."""
        c = self.corr
        return c[0] + (np.asarray(q, float) - c[1]) / c[2]

    def layout_point(self, raw):
        """A point in raw registered board coords -> metres on the layout map (clamped to the room), or None."""
        if self.model is None:
            return None
        x0, sp, y0, pitch = self.model
        b = self.fix(raw)
        fx, fy = (float(b[0]) - x0) / sp, (float(b[1]) - y0) / pitch
        x = float(self._interp(fx, self.cols_x)); y = float(self._interp(fy, self.rows_y[0]))
        W, Hh = float(self.layout.get("width_m", 8)), float(self.layout.get("height_m", 6))
        return [round(min(max(x, 0.0), W), 3), round(min(max(y, 0.0), Hh), 3)]

    def take_events(self):
        e, self.events = self.events, []
        return e

    # ---------- placement ----------
    def _fit(self, tags):
        """Grid model: x = x0 + spacing * segment_index, y = y0 + pitch * row_index (board coordinates)."""
        wm = float(np.median([t.w for t in tags])); hm = float(np.median([t.h for t in tags]))
        anc = [(t, self.vocab[t.id]) for t in tags if t.known and self.vocab.get(t.id)]
        if anc:
            sp = [(a.pos[0] - b.pos[0]) / (sa - sb) for i, (a, (sa, _)) in enumerate(anc) for b, (sb, _) in anc[:i] if sa != sb]
            spacing = float(np.median(sp)) if sp else SPACING_PER_TEXT_W * wm
            x0 = float(np.median([t.pos[0] - spacing * si for t, (si, _) in anc]))
            pp = [(a.pos[1] - b.pos[1]) / (ra - rb) for i, (a, (sa, ra)) in enumerate(anc) for b, (sb, rb) in anc[:i] if sa == sb and ra != rb]
            if not pp:
                pp = [(a.pos[1] - b.pos[1]) / (ra - rb) for i, (a, (_, ra)) in enumerate(anc) for b, (_, rb) in anc[:i] if ra != rb]
            pitch = float(np.median(pp)) if pp else PITCH_PER_TEXT_H * hm
            if pitch < 1.5 * hm:                                 # nonsense (tags moved onto one row): use the text size
                pitch = PITCH_PER_TEXT_H * hm
            y0 = float(np.median([t.pos[1] - pitch * ri for t, (_, ri) in anc]))
            return x0, spacing, y0, pitch
        # no known tags: cluster columns by gap, rows by the smallest vertical gap
        xs = sorted(t.pos[0] for t in tags)
        cl = [[xs[0]]]
        for x in xs[1:]:
            (cl[-1].append(x) if x - cl[-1][-1] < 2.4 * wm else cl.append([x]))
        cen = [float(np.mean(c)) for c in cl]
        spacing = float(np.median(np.diff(cen))) if len(cen) > 1 else SPACING_PER_TEXT_W * wm
        gaps = []
        for c in cl:
            ys = sorted(t.pos[1] for t in tags if t.pos[0] in c or abs(t.pos[0] - np.mean(c)) < 1.2 * wm)
            gaps += [b - a for a, b in zip(ys, ys[1:]) if b - a > 1.5 * hm]
        pitch = float(np.median([g for g in gaps if g < 1.4 * min(gaps)])) if gaps else PITCH_PER_TEXT_H * hm
        return cen[0], spacing, min(t.pos[1] for t in tags), pitch

    @staticmethod
    def _snap(f, prev, n):
        i = prev if prev is not None and abs(f - prev) < .62 else int(round(f))     # hysteresis: no flicker at boundaries
        return max(0, min(n - 1, i))

    def containers(self):
        tags = [t for t in self.tags.values() if t.n >= CONFIRM]
        if not tags:
            self.model = None
            return []
        self.model = x0, sp, y0, pitch = self._fit(tags)
        anc = [(t, self.vocab[t.id]) for t in tags if t.known and self.vocab.get(t.id)]
        if len(anc) >= 3:                                        # known tags must sit on a regular column grid, else the geometry is not settled yet
            good = sum(abs((t.pos[0] - x0) / sp - si) < .3 for t, (si, _) in anc)
            if good < .8 * len(anc):
                return self._last_rows
        by_seg = {}
        for t in tags:                                           # column -> segment
            t.si = self._snap((t.pos[0] - x0) / sp, t.si, len(self.segs))
            by_seg.setdefault(t.si, []).append(t)
        out = []
        for si, ts in by_seg.items():
            zone, g = self.segs[si]
            ys, nrows = self.rows_y[si], len(self.rows_y[si])
            ts.sort(key=lambda t: t.pos[1])                       # top first
            # tags the video confirms in their declared place keep the declared slot (layout.json);
            # if the order in the column changed, or the tag is new / came from another column, the slot follows the video
            known = [t for t in ts if self.vocab.get(t.id) and self.vocab[t.id][0] == si]
            krows = [self.vocab[t.id][1] for t in known]
            row_of, used = {}, set()
            if all(a < b for a, b in zip(krows, krows[1:])):
                for t, r in zip(known, krows):
                    row_of[t.id] = min(r, nrows - 1); used.add(row_of[t.id])
            prev = -1
            for t in ts:
                if t.id not in row_of:
                    r = prev + 1
                    while r in used and r < nrows - 1:
                        r += 1
                    row_of[t.id] = min(r, nrows - 1); used.add(row_of[t.id])
                prev = row_of[t.id]
            if STACK_COLUMNS:                                     # every column is one stack: bottom box = tier 1
                base = max(row_of[t.id] for t in ts)
                cells = {base: sorted(ts, key=lambda t: -t.pos[1])}
            else:
                cells = {}
                for t in ts:
                    cells.setdefault(row_of[t.id], []).append(t)
                cells = {r: sorted(c, key=lambda t: t.first) for r, c in cells.items()}
            for r, cell in cells.items():
                for k, t in enumerate(cell, 1):
                    out.append({"container_id": t.id[:11], "checkpoint": f"{zone}/{g['name']}",
                                "x": round(self.cols_x[si], 2), "y": round(ys[r], 2), "z_tier": k,
                                "z_m": round(k * CONTAINER_H_M, 2), "heading_deg": 90.0, "heading_ambiguous": True,
                                "ocr_conf": round(t.conf, 2), "seen_at": None, "in_view": bool(t.in_view), "live": True,
                                "reads": t.n})
        self._last_rows = sorted(out, key=lambda c: (c["checkpoint"], -c["y"], c["z_tier"]))
        return self._last_rows

    def view_polygon(self):
        """Camera footprint on the layout map (metres), or None."""
        if self.foot is None or self.model is None:
            return None
        x0, sp, y0, pitch = self.model
        pts = []
        for (bx, by) in self.foot.reshape(-1, 2):
            fx, fy = (bx - x0) / sp, (by - y0) / pitch
            pts.append([round(float(self._interp(float(fx), self.cols_x)), 2), round(float(self._interp(float(fy), self.rows_y[0])), 2)])
        return pts

    @staticmethod
    def _interp(f, vals):
        """vals indexed 0..n-1; linear inside, extrapolated with the end spacing."""
        n = len(vals)
        if n == 1:
            return vals[0]
        if f <= 0:
            return vals[0] + f * (vals[1] - vals[0])
        if f >= n - 1:
            return vals[-1] + (f - (n - 1)) * (vals[-1] - vals[-2])
        i = int(f)
        return vals[i] + (f - i) * (vals[i + 1] - vals[i])

    def overlay(self):
        """Tags for drawing on the video, in the RAW registered coordinates (the ones a frame's transform H maps
        pixels to). A tag read a moment ago is drawn where it was measured, so the box sits exactly on its text;
        otherwise the stored (drift-corrected) position is converted back to raw coordinates."""
        s = float(self.corr[2]) or 1.0
        out = []
        for t in self.tags.values():
            if t.raw is not None and self.step - t.raw[3] <= RAW_AGE:
                pos, w, h = t.raw[0], t.raw[1], t.raw[2]
            else:
                pos, w, h = self.unfix(t.pos), t.w / s, t.h / s
            out.append((t.id, pos, w, h, t.n >= CONFIRM, t.in_view))
        return out