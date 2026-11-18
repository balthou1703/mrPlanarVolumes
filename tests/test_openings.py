"""Openings: a synthetic recess map of a wall with an arched door crossed by the camera, a rectangular window and a
round window -> primitives of the right shape, the door goes through the wall, the slab stays light."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from planarVolumes import openings  # noqa: E402
from planarVolumes.recesses import CELL  # noqa: E402

F, S, U = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
RECT = (0.0, 12.0, 0.0, 8.0)                      # b (along S), a (along U)


class PathScene:
    """Minimal scene: camera centers along a path crossing the wall plane F = 0 at (b, a) = (3, 1.2)."""

    def __init__(self):
        self.path = np.array([[5.05 - 0.2 * i, 3.0, 1.2] for i in range(40)])

    def centers(self, frames=None):
        return self.path if frames is None else self.path[list(frames)]


def recessMap():
    nb, na = int((RECT[1] - RECT[0]) / CELL), int((RECT[3] - RECT[2]) / CELL)
    bb = RECT[0] + (np.arange(nb) + .5) * CELL
    aa = RECT[2] + (np.arange(na) + .5) * CELL
    B, A = np.meshgrid(bb, aa)
    med = np.zeros((na, nb), np.float32)
    door = openings.rasterize(dict(type="arch", x0=2.0, x1=4.0, y0=0.0, ys=2.2, rho=1.0), bb, aa)
    med[door] = 0.4                                  # recessed (open door seen at an angle: depth not large)
    window = (B > 7.0) & (B < 8.5) & (A > 1.0) & (A < 3.0)
    med[window] = 0.3
    oculus = (B - 10.0) ** 2 + (A - 6.0) ** 2 < 0.6 ** 2
    med[oculus] = 0.25
    rng = np.random.default_rng(0)
    med += rng.normal(0, 0.01, med.shape).astype(np.float32)
    return dict(med=med, cell=CELL, rect=RECT, frames=list(range(10)), off=0.0, axis="F")


def large():
    return dict(frame=dict(F=F, S=S, U=U), planes=[dict(axis="F", off=0.0, rect=list(RECT), dom="exterior", support=50.0,
                                                        cuts=[])])


def test_wall_openings_shapes_and_through_door():
    objects, P, rect = openings.wallOpenings(PathScene(), "exterior_F0.00", large(), recessMap(), cameraSide=1.0,
                                             log=lambda s: None)
    types = sorted(p["type"] for p in P)
    assert types == ["arch", "circle", "rect"], types
    door = next(p for p in P if p["type"] == "arch")
    assert abs(door["x0"] - 2.0) < 0.15 and abs(door["x1"] - 4.0) < 0.15
    assert abs(door["ys"] + door["rho"] - 3.2) < 0.15
    assert door["y0"] == RECT[2]                      # down to the floor
    assert door["through"] and door.get("crossed")    # crossed by the camera: through, no glazing
    names = [n for n, _ in objects]
    assert "wall" in names
    assert sum(n.startswith("glazing") for n in names) == 2
    window = next(p for p in P if p["type"] == "rect")
    assert abs(window["x0"] - 7.0) < 0.15 and abs(window["x1"] - 8.5) < 0.15


def test_holed_slab_area():
    P = [dict(type="rect", x0=2.0, x1=4.0, y0=1.0, ys=3.0, rho=0.0)]
    polys = openings.holedSlab(RECT, P)
    area = 0.0
    for poly in polys:
        x, y = poly[:, 0], poly[:, 1]
        area += 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    assert np.isclose(area, 12 * 8 - 4)
    assert all(len(p) == 4 for p in polys)


def test_arch_spandrels_fill_the_box():
    P = [dict(type="arch", x0=2.0, x1=4.0, y0=0.0, ys=2.0, rho=1.0)]
    polys = openings.holedSlab(RECT, P)
    area = 0.0
    for poly in polys:
        x, y = poly[:, 0], poly[:, 1]
        area += 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    hole = 2.0 * 2.0 + 0.5 * np.pi * 1.0 ** 2          # rectangle + half disc
    assert abs(area - (12 * 8 - hole)) < 0.05


def test_regularize_aligns_sills_and_widths():
    P = [dict(type="rect", x0=0.0, x1=1.0, y0=1.0, ys=2.0, rho=0.0),
         dict(type="rect", x0=3.0, x1=4.2, y0=1.2, ys=2.1, rho=0.0)]
    openings.regularize(P)
    assert P[0]["y0"] == P[1]["y0"]
    assert np.isclose(P[0]["x1"] - P[0]["x0"], P[1]["x1"] - P[1]["x0"])
