"""Glue: per-frame tracker output -> one validated sighting per container.

Wire `Detection` to your base repo (container-id-detection): for every tracked
container per frame, fill track_id, the OBB, the OCR text read from that track,
and optionally the pixel position of the door end.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

import db
import geometry as geo
from iso6346 import clean_ocr


@dataclass
class Detection:
    track_id: int
    cx: float; cy: float; w: float; h: float; angle_rad: float   # OBB, pixels
    ocr_text: str | None = None          # raw OCR for this frame (may be None)
    ocr_conf: float = 0.0
    door_px: tuple | None = None         # pixel of the door end, if known
    top_height_m: float | None = None    # from depth/LiDAR; None -> tier 1


class CheckpointPipeline:
    def __init__(self, checkpoint: str, homography: geo.Homography, min_votes: int = 3):
        self.checkpoint = checkpoint
        self.H = homography
        self.min_votes = min_votes
        self.votes = defaultdict(Counter)   # track_id -> Counter of valid IDs
        self.saved = set()                  # track_ids already written

    def update(self, detections: list[Detection]) -> list[int]:
        """Call once per frame. Returns ids of sightings written this frame."""
        written = []
        for d in detections:
            if d.track_id in self.saved:
                continue
            if d.ocr_text:
                cid = clean_ocr(d.ocr_text)       # check-digit validated
                if cid:
                    self.votes[d.track_id][cid] += 1
            if not self.votes[d.track_id]:
                continue
            cid, n = self.votes[d.track_id].most_common(1)[0]
            if n < self.min_votes:
                continue
            x, y = self.H.to_ground(d.cx, d.cy)
            hdg, amb = geo.heading_deg(self.H, d.cx, d.cy, d.w, d.h, d.angle_rad, d.door_px)
            tier = geo.stack_tier(d.top_height_m) if d.top_height_m else 1
            written.append(db.add_sighting(
                container_id=cid, checkpoint=self.checkpoint, x=x, y=y,
                z_tier=tier, heading_deg=hdg, heading_ambiguous=amb,
                ocr_conf=d.ocr_conf))
            self.saved.add(d.track_id)
        return written
