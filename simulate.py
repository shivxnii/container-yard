"""Fake live feed: python simulate.py  (leave running, open the dashboard)."""
import math, random, time
import db
from iso6346 import check_digit

db.init_db()
OWNERS = ["MSKU", "CSQU", "TGHU", "HLXU", "OOLU"]
CHECKPOINTS = ["Gate-1", "Crane-A", "Block-C", "Port-B"]


def make_id():
    base = random.choice(OWNERS) + f"{random.randint(0, 999999):06d}"
    return base + str(check_digit(base))


fleet = [{"id": make_id(), "x": random.uniform(10, 90), "y": random.uniform(10, 90),
          "h": random.uniform(0, 360), "tier": random.randint(1, 3)} for _ in range(8)]
while True:
    c = random.choice(fleet)
    c["h"] = (c["h"] + random.choice([0, 0, 90, -90, 180])) % 360   # direction change
    r = random.uniform(2, 8)
    c["x"] = min(95, max(5, c["x"] + r * math.sin(math.radians(c["h"]))))
    c["y"] = min(95, max(5, c["y"] + r * math.cos(math.radians(c["h"]))))
    db.add_sighting(container_id=c["id"], checkpoint=random.choice(CHECKPOINTS),
                    x=c["x"], y=c["y"], z_tier=c["tier"], heading_deg=c["h"],
                    heading_ambiguous=random.random() < 0.2,
                    ocr_conf=round(random.uniform(.85, .99), 2))
    print("sighting", c["id"], round(c["x"]), round(c["y"]), round(c["h"]))
    time.sleep(1.5)
