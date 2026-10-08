import os, math
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
import db, geometry as geo
from iso6346 import is_valid, clean_ocr, check_digit
from pipeline import CheckpointPipeline, Detection


def test_iso():
    assert check_digit("CSQU305438") == 3          # worked example in the standard
    assert is_valid("CSQU3054383")
    assert not is_valid("CSQU3054384")
    assert clean_ocr("xx CSQU 3O54383 yy") == "CSQU3054383"   # O->0 fixed
    assert clean_ocr("garbage") is None


def make_H():
    # image 1000x1000 px <-> 100x100 m yard, y flipped (image y goes down)
    px = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    gd = [(0, 100), (100, 100), (100, 0), (0, 0)]
    return geo.Homography(px, gd)


def test_geometry():
    H = make_H()
    x, y = H.to_ground(500, 500)
    assert abs(x - 50) < 1e-6 and abs(y - 50) < 1e-6
    # long axis pointing right in the image = east = 90 deg
    hdg, amb = geo.heading_deg(H, 500, 500, 120, 30, 0.0)
    assert abs(hdg % 180 - 90) < 1e-6 and amb
    # door on the left side -> heading points west (270)
    hdg, amb = geo.heading_deg(H, 500, 500, 120, 30, 0.0, door_px=(400, 500))
    assert abs(hdg - 270) < 1e-6 and not amb
    assert geo.stack_tier(2.59) == 1 and geo.stack_tier(7.8) == 3


def test_pipeline_and_db():
    db.init_db()
    p = CheckpointPipeline("Gate-1", make_H(), min_votes=2)
    det = lambda t: Detection(7, 500, 500, 120, 30, 0.0, ocr_text=t, ocr_conf=.9,
                              door_px=(400, 500))
    assert p.update([det("CSQU3O54383")]) == []            # 1 vote, not enough
    assert p.update([det("CSQU3054383")]) != []            # 2 votes -> saved
    assert p.update([det("CSQU3054383")]) == []            # not saved twice
    r = db.latest("CSQU3054383")
    assert r.checkpoint == "Gate-1" and abs(r.x - 50) < 1e-6 and r.z_tier == 1
    assert len(db.history("CSQU3054383")) == 1
