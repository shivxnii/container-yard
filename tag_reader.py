"""Reads the handwritten tag codes (#LQ6872, #S43GLN, ...) from a video frame.

Engine: RapidOCR (pip install rapidocr-onnxruntime). Pure pip, no Tesseract, runs on CPU, models are bundled.
Noisy reads ("#SY3GLN", "#25966L", "#FGQIS4") are snapped to the known tags of layout.json (or tags.txt) with a
letter-confusion-aware edit distance (S/5, O/0, 4/Y, M/H, ...). Unknown tags are still accepted when they look
like a tag ("#" + 5-8 letters/digits), so a new tag written on the board shows up as it was read.
"""
import re
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

_engine, _error = None, None

# characters that handwriting / OCR mix up: substituting one for the other is cheap
_GROUPS = ["S5", "O0DQ", "I1L", "Z2", "B8", "G6", "Y4A", "14", "MHN", "WVU", "4Q9", "T7", "PR", "E3", "CG", "KX"]
_SIM = {(a, b) for g in _GROUPS for a in g for b in g if a != b}
SNAP_MAX = 0.37          # accepted cost per character
SNAP_MARGIN = 0.06       # best match must beat the runner-up by this much (else ambiguous)


def load_engine():
    """Load the OCR engine once. Returns (engine, error_message)."""
    global _engine, _error
    if _engine is None and _error is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _engine = RapidOCR()
        except Exception as e:                                  # not installed / broken install
            _error = ("OCR engine missing. Run:  pip install rapidocr-onnxruntime   "
                      f"({type(e).__name__}: {e})")
    return _engine, _error


def norm(text):
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def wdist(a, b):
    """Edit distance where look-alike characters cost 0.35 instead of 1."""
    prev = [float(j) for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [float(i)]
        for j, cb in enumerate(b, 1):
            sub = 0.0 if ca == cb else (0.35 if (ca, cb) in _SIM else 1.0)
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub))
        prev = cur
    return prev[-1]


def snap(text, vocab):
    """-> (tag, score 0..1) of the closest known tag, or None. vocab: iterable of known tags."""
    c = norm(text)
    if len(c) < 4 or not vocab:
        return None
    sc = sorted((wdist(c, v) / max(len(c), len(v)), v) for v in vocab)
    best = sc[0]
    if best[0] > SNAP_MAX:
        return None
    if len(sc) > 1 and sc[1][0] - best[0] < SNAP_MARGIN:
        return None                                              # two known tags fit equally well
    return best[1], round(1 - best[0], 2)


def looks_like_tag(raw):
    c = norm(raw)
    return ("#" in str(raw) and 5 <= len(c) <= 8 and sum(ch.isdigit() for ch in c) >= 2
            and sum(ch.isalpha() for ch in c) >= 2)


def parse(items, vocab, min_conf=0.5):
    """OCR items [(text, conf, cx, cy, w, h)] -> tag reads + positions of any tag-like text.
    Returns (reads, seen): reads = dicts with id/known/conf/cx/cy/w/h, seen = [(cx, cy)] of all tag-like text."""
    reads, seen = {}, []
    for text, conf, cx, cy, w, h in items:
        if conf < min_conf:
            continue
        s = snap(text, vocab)
        if s:
            tid, q = s
            known = True
        elif looks_like_tag(text) and conf >= 0.7:
            tid, q, known = norm(text), 0.5, False
        else:
            continue
        seen.append((cx, cy))
        score = conf * q
        if tid not in reads or reads[tid]["score"] < score:      # same tag twice in one frame: keep the better read
            reads[tid] = {"id": tid, "known": known, "conf": round(conf, 2), "score": score,
                          "raw": str(text), "cx": cx, "cy": cy, "w": w, "h": h}
    return list(reads.values()), seen


def read_frame(frame, width=1280):
    """Run OCR on a BGR frame. Returns ([(text, conf, cx, cy, w, h)], (frame_w, frame_h)) in the pixels of the resized frame."""
    eng, err = load_engine()
    if eng is None:
        raise RuntimeError(err)
    if frame.shape[1] > width:
        frame = cv2.resize(frame, None, fx=width / frame.shape[1], fy=width / frame.shape[1])
    res, _ = eng(frame)
    items = []
    for box, text, conf in (res or []):
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        items.append((text, float(conf), (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                      max(xs) - min(xs), max(ys) - min(ys)))
    return items, (frame.shape[1], frame.shape[0])


def vocab_from_layout(layout, extra_file=None):
    """{tag: (segment_index, row_index)} from layout.json (+ optional tags.txt, one tag per line, no position)."""
    v, si = {}, 0
    for z in layout["zones"]:
        for g in z["segments"]:
            rows = sorted(g["slots"], key=lambda s: -s["y"])
            for ri, sl in enumerate(rows):
                if sl.get("tag"):
                    v[norm(sl["tag"])] = (si, ri)
            si += 1
    if extra_file and Path(extra_file).exists():
        for line in Path(extra_file).read_text().splitlines():
            if norm(line):
                v.setdefault(norm(line), None)
    return v


if __name__ == "__main__":
    lay = {"zones": [{"segments": [{"slots": [{"y": 1, "tag": "LQ6872"}, {"y": 2, "tag": "S43NOY"}]}, {"slots": [
        {"y": 1, "tag": "LPZ126"}, {"y": 2, "tag": "S43GLN"}, {"y": 3, "tag": "GKPS97"}, {"y": 4, "tag": "LMW440"},
        {"y": 5, "tag": "PQR567"}, {"y": 6, "tag": "FGQ154"}, {"y": 7, "tag": "2546GL"}, {"y": 8, "tag": "123FM15"}]}]}]}
    V = vocab_from_layout(lay)
    cases = {"#SY3GLN": "S43GLN", "#25966L": "2546GL", "#FGQIS4": "FGQ154", "#LWq40": "LMW440", "#123FH5": "123FM15",
             "#P2126": "LPZ126", "#5130y": "S43NOY", "#LQG872": "LQ6872", "#6KPS97": "GKPS97", "#PQRS67": "PQR567",
             "Lmwqqo": "LMW440", "#2596GL": "2546GL", "#GKps97": "GKPS97", "#LP2126": "LPZ126", "#51240y": "S43NOY",
             "ZONE-A": None, "Segment-1": None, "Help Conserve": None, "#L812": None}
    bad = 0
    for raw, want in cases.items():
        got = snap(raw, V); got = got[0] if got else None
        bad += got != want
        print(("ok  " if got == want else "FAIL"), raw, "->", got, "(want", want, ")")
    print("failures:", bad)