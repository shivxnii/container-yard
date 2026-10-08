"""Live video source -> MJPEG stream. Optional: needs opencv-python.

VIDEO_SOURCE = RTSP url, video file path, or 0 for a webcam.
Set `feed.annotate = fn(frame) -> frame` to draw your detections/IDs on frames.
"""
import os, threading, time
try:
    import cv2
except ImportError:          # dashboard still works without video
    cv2 = None

annotate = None
_latest = None
_lock = threading.Lock()


def start(source=None):
    source = source if source is not None else os.getenv("VIDEO_SOURCE")
    if cv2 is None or source in (None, ""):
        return False
    src = int(source) if str(source).isdigit() else source
    threading.Thread(target=_loop, args=(src,), daemon=True).start()
    return True


def _loop(src):
    global _latest
    cap = cv2.VideoCapture(src)
    while True:
        ok, frame = cap.read()
        if not ok:                       # file ended / stream dropped: retry
            time.sleep(1); cap.release(); cap = cv2.VideoCapture(src); continue
        if annotate:
            frame = annotate(frame)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with _lock:
                _latest = jpg.tobytes()
        time.sleep(0.03)


def available():
    return _latest is not None


def mjpeg():
    while True:
        with _lock:
            jpg = _latest
        if jpg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.05)
