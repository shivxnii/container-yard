# Container tracker – live dashboard

    pip install -r requirements.txt
    python simulate.py                 # terminal 1: fake live sightings (skip once real data flows)
    uvicorn api:app --reload           # terminal 2
    open http://localhost:8000

Live camera: `VIDEO_SOURCE=rtsp://user:pass@cam/stream uvicorn api:app` (or a file path, or 0 for a webcam).
Draw detections on the stream by setting `feed.annotate = fn(frame) -> frame`.
Real data: run `pipeline.CheckpointPipeline.update(...)` in another process with the same `DATABASE_URL`;
the dashboard picks new sightings up within 2 seconds. PostgreSQL: `DATABASE_URL=postgresql+psycopg2://...`

## Room mode (whiteboard layout)
    ROOM_MODE=1 uvicorn api:app --reload
    python simulate_room.py          # tags move between slots (demo)
Layout (room size, zones, segments, slots) is in `layout.json`. ROOM_MODE=1 also skips ISO 6346 validation,
because the whiteboard tags (#PQR567 ...) are not valid container numbers. Unset it for the real yard.

## Video upload + z
Dashboard (room mode) has an **Upload video** panel: boxes are found, mapped to zone / segment / slot and written as sightings.
Needs `pip install opencv-python`. Handheld pans work (frames are registered to each other); a big .MOV is decoded on the server,
so H.264 .mp4 is faster. `OCR=1` (+ pytesseract + Tesseract) tries to read the tags, else ids are `V<job>-B<nn>`.
z: `z_tier` (1 = ground) and `z_m` = top height = tier x 2.59 m. In video, boxes in one column that nearly touch count as a stack
(`Z_GAP`, default 0.45 x box height; `Z_MODE=flat` turns it off). Simulator stacks up to 3 high.
The baseline detector in `video_processor.find_boxes` is classical OpenCV; replace it with your own detector for accuracy.
## Live simulation (map follows the video / camera)
    pip install -r requirements.txt           # includes opencv-python and rapidocr-onnxruntime (reads the handwritten tags)
    # PowerShell
    $env:ROOM_MODE=1; uvicorn api:app --reload                          # then use the Live simulation panel
    $env:ROOM_MODE=1; $env:VIDEO_SOURCE=0; uvicorn api:app --reload      # or: webcam drives the map from startup
Dashboard (room mode) -> **Live simulation**: *Play video* (file is played in real time) or *Start camera* (0/1/2 or rtsp:// URL).
How it works: the tag codes (#LQ6872 ...) are read with OCR, snapped to the tags in `layout.json` (or `tags.txt`, one tag per line),
placed with the camera motion tracked, and drawn on the map. A tag that is rewritten in another column / order moves on the map,
a tag that is wiped off the board disappears. Every change is saved as a sighting. A video file waits for the OCR if the PC is slow,
so no tag is skipped (the picture holds for a moment). `STACK_COLUMNS=1` shows each board column as one stack (z) instead of one box per slot.
Other env: `OCR_WIDTH=1280` (bigger reads smaller text, slower), `LIVE_STEP_S=0.35`, `CAMERA_FIXED=1` (camera does not move),
`LIVE_HWACCEL=0` (no hardware video decoding). `simulate_room.py` is no longer needed.