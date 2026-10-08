"""Pixel -> yard coordinates, heading and stack level."""
import math
import numpy as np

CONTAINER_HEIGHT_M = 2.59   # standard 8'6" container


class Homography:
    """Maps image pixels to ground-plane metres (yard frame: x east, y north)."""

    def __init__(self, pixel_pts, ground_pts):
        assert len(pixel_pts) >= 4 and len(pixel_pts) == len(ground_pts)
        rows = []
        for (u, v), (x, y) in zip(pixel_pts, ground_pts):
            rows.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
            rows.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])
        _, _, vt = np.linalg.svd(np.array(rows, dtype=float))
        h = vt[-1].reshape(3, 3)
        self.H = h / h[2, 2]

    def to_ground(self, u, v):
        p = self.H @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])


def heading_deg(H: Homography, cx, cy, w, h, angle_rad, door_px=None):
    """Heading of the container's long axis, degrees clockwise from north (+y).

    OBB gives the axis only (0-180 deg). If door_px (pixel location of the
    door end) is given, the 180-degree ambiguity is resolved and the heading
    points toward the door end. Returns (heading, ambiguous).
    """
    ax = (math.cos(angle_rad), math.sin(angle_rad)) if w >= h else \
         (-math.sin(angle_rad), math.cos(angle_rad))
    L = max(w, h) / 2
    x0, y0 = H.to_ground(cx, cy)
    x1, y1 = H.to_ground(cx + ax[0] * L, cy + ax[1] * L)
    dx, dy = x1 - x0, y1 - y0
    ambiguous = True
    if door_px is not None:
        gx, gy = H.to_ground(*door_px)
        if (gx - x0) * dx + (gy - y0) * dy < 0:
            dx, dy = -dx, -dy
        ambiguous = False
    return math.degrees(math.atan2(dx, dy)) % 360, ambiguous


def stack_tier(height_above_ground_m: float) -> int:
    """Tier 1 = on the ground. height = top of container above ground."""
    return max(1, round(height_above_ground_m / CONTAINER_HEIGHT_M))
