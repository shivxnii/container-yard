"""Run:  VIDEO_SOURCE=rtsp://... uvicorn api:app --reload   then open http://localhost:8000
Room mode + live camera:   ROOM_MODE=1 VIDEO_SOURCE=0 uvicorn api:app      (map follows the camera)
Room mode, no env source:  ROOM_MODE=1 uvicorn api:app    (start a video / camera from the dashboard)"""
import json, os, threading, time, uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
import db, feed
import live_session as live
import geometry as geo
from iso6346 import is_valid

ROOM_MODE = os.getenv("ROOM_MODE") == "1"   # whiteboard/room demo: tags are not ISO 6346
app = FastAPI(title="Container tracker")
db.init_db()
_src = os.getenv("VIDEO_SOURCE")
if ROOM_MODE and _src and Path(__file__).with_name("layout.json").exists():
    # room mode: the camera drives the map (one reader for the camera, so feed.py is not started)
    db.clear_all()
    live.start(int(_src) if _src.isdigit() else _src, json.loads(Path(__file__).with_name("layout.json").read_text()))
else:
    feed.start()


def _row(r):
    return {"container_id": r.container_id, "checkpoint": r.checkpoint,
            "x": r.x, "y": r.y, "z_tier": r.z_tier, "z_m": round(r.z_tier * geo.CONTAINER_HEIGHT_M, 2),
            "heading_deg": r.heading_deg, "heading_ambiguous": r.heading_ambiguous,
            "ocr_conf": r.ocr_conf,
            "seen_at": r.seen_at.isoformat() if r.seen_at else None}


@app.get("/")
def dashboard():
    return FileResponse(Path(__file__).with_name("dashboard.html"))


@app.get("/containers")
def all_containers():
    return [_row(r) for r in db.latest_all()]


@app.get("/sightings/recent")
def recent(limit: int = 30):
    return [_row(r) for r in db.recent(min(limit, 200))]


@app.get("/containers/{container_id}/latest")
def where_is(container_id: str):
    cid = container_id.upper()
    if not ROOM_MODE and not is_valid(cid):
        raise HTTPException(422, "Invalid ISO 6346 number")
    r = db.latest(cid)
    if not r:
        raise HTTPException(404, "Container not seen yet")
    return _row(r)


@app.get("/containers/{container_id}/history")
def route(container_id: str):
    return [_row(r) for r in db.history(container_id.upper())]


@app.get("/video")
def video():
    if not feed.available():
        raise HTTPException(404, "No live video source")
    return StreamingResponse(feed.mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/layout")
def layout():
    """Room/yard layout (zones, segments, slots). 404 -> dashboard uses the plain 100 m grid."""
    p = Path(__file__).with_name("layout.json")
    if not ROOM_MODE or not p.exists():
        raise HTTPException(404, "No layout")
    return json.loads(p.read_text())


# ---------- video upload -> zone / segment output ----------
UPLOADS = Path(__file__).with_name("uploads"); UPLOADS.mkdir(exist_ok=True)
jobs, job_count = {}, 0


def _run(job_id, path, name):
    j = jobs[job_id]
    try:
        import video_processor as vp
        lay = json.loads(Path(__file__).with_name("layout.json").read_text())
        rows = vp.process(str(path), lay, j["no"],
                          progress=lambda p, m="": j.update(progress=round(p, 2), message=m),
                          preview_path=str(UPLOADS / f"{job_id}.jpg"))
        placed = [r for r in rows if r["segment"] != "Unassigned"]
        db.clear_all()                                   # the processed video replaces what was on the map
        for r in placed:
            db.add_sighting(container_id=r["container_id"], checkpoint=f'{r["zone"]}/{r["segment"]}',
                            x=r["x"], y=r["y"], z_tier=r["z_tier"], heading_deg=90.0,
                            heading_ambiguous=True, ocr_conf=0.5)
        summary = {}
        for r in rows:
            k = f'{r["zone"]}/{r["segment"]}'
            s = summary.setdefault(k, {"count": 0, "max_tier": 0})
            s["count"] += 1; s["max_tier"] = max(s["max_tier"], r["z_tier"])
        j.update(status="done", progress=1, message="Done", rows=placed, summary=summary,
                 unassigned=len(rows) - len(placed), preview=(UPLOADS / f"{job_id}.jpg").exists())
    except Exception as e:
        j.update(status="error", message=str(e))


@app.post("/upload")
async def upload(request: Request, name: str = "video.mp4"):
    """Raw video bytes in the request body (no multipart dependency)."""
    global job_count
    if not Path(__file__).with_name("layout.json").exists():
        raise HTTPException(404, "layout.json missing")
    job_id = uuid.uuid4().hex[:8]; job_count += 1
    path = UPLOADS / f"{job_id}{Path(name).suffix[:5] or '.mp4'}"
    with open(path, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    jobs[job_id] = {"id": job_id, "no": job_count, "name": name, "status": "running", "progress": 0, "message": "Queued"}
    threading.Thread(target=_run, args=(job_id, path, name), daemon=True).start()
    return jobs[job_id]


@app.get("/jobs/{job_id}")
def job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[job_id]


@app.get("/jobs/{job_id}/preview")
def job_preview(job_id: str):
    p = UPLOADS / f"{job_id}.jpg"
    if not p.exists():
        raise HTTPException(404, "No preview")
    return FileResponse(p)


# ---------- live simulation: video file or camera drives the map ----------
def _layout_json():
    p = Path(__file__).with_name("layout.json")
    if not p.exists():
        raise HTTPException(404, "layout.json missing")
    return json.loads(p.read_text())


@app.post("/live/start")
def live_start(source: str):
    """source = camera index (0, 1, ...), an rtsp/http URL, or a video file path on the server."""
    if live.cv2 is None:
        raise HTTPException(500, "opencv-python is not installed:  pip install opencv-python")
    src = int(source) if source.isdigit() else source
    if isinstance(src, str) and not src.startswith(("rtsp://", "rtmp://", "http://", "https://")) and not Path(src).is_file():
        raise HTTPException(404, f"File not found: {src}")
    layout = _layout_json()
    db.clear_all()                                       # empty map: it is rebuilt from this source only
    return live.start(src, layout).snapshot()


@app.post("/live/upload")
async def live_upload(request: Request, name: str = "video.mp4"):
    """Raw video bytes in the body; saved, then played in real time through the live simulation."""
    layout = _layout_json()
    path = UPLOADS / f"live_{uuid.uuid4().hex[:6]}{Path(name).suffix[:5] or '.mp4'}"
    with open(path, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    db.clear_all()
    return live.start(str(path), layout).snapshot()


@app.post("/live/stop")
def live_stop():
    live.stop()
    return live.state()


@app.post("/reset")
def reset():
    """Clear the map: stop any live session and delete all stored sightings."""
    live.reset()
    db.clear_all()
    return {"ok": True}


@app.get("/live/state")
def live_state():
    return live.state()


@app.get("/live/crop/{tag}")
def live_crop(tag: str):
    """Picture of one tag / container cut out of the newest live frame."""
    jpg = live.crop(tag)
    if jpg is None:
        raise HTTPException(404, "Not in the picture right now")
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/live/video")
def live_video():
    return StreamingResponse(live.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")