"""Room demo: tags from the whiteboard move between slots, and can stack (z up to 3).
Run:  ROOM_MODE=1 uvicorn api:app --reload   and   python simulate_room.py
Replace this with real detections: CheckpointPipeline.update(...) writes the same rows."""
import json, random, time
from pathlib import Path
import db

db.init_db()
L = json.loads((Path(__file__).with_name("layout.json")).read_text())
slots = [(z["name"], s["name"], sl) for z in L["zones"] for s in z["segments"] for sl in s["slots"]]


def put(tag, i, heading=90.0, tier=1):
    zone, seg, sl = slots[i]
    db.add_sighting(container_id=tag, checkpoint=f"{zone}/{seg}", x=sl["x"], y=sl["y"],
                    z_tier=tier, heading_deg=heading, heading_ambiguous=False, ocr_conf=0.9)
    print(tag, "->", zone, seg, "tier", tier)


stacks = {i: [sl["tag"]] for i, (_, _, sl) in enumerate(slots) if sl["tag"]}   # slot -> tags, bottom first
for i, st in stacks.items():             # initial detections
    for t, tag in enumerate(st, 1):
        put(tag, i, tier=t)
while True:
    time.sleep(2.5)
    i = random.choice(list(stacks)); tag = stacks[i].pop()          # only the top container moves
    if not stacks[i]:
        del stacks[i]
    empty = [k for k in range(len(slots)) if k not in stacks]
    stackable = [k for k in stacks if len(stacks[k]) < 3 and k != i]
    j = random.choice(stackable) if stackable and random.random() < .35 else random.choice(empty)
    stacks.setdefault(j, []).append(tag)
    put(tag, j, random.choice([90.0, 270.0]), tier=len(stacks[j]))