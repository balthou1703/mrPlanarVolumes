"""Large planes: ghost walls (foliage in front of a facade) removed, walls extended along the dense surface up to the
corner of a perpendicular wall, floors found under the camera path (and rule-based levels moved to their height)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from planarVolumes.largePlanes import LargePlanes  # noqa: E402

F, S, U = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])


class AxisFrame:
    axes = {"F": F, "S": S, "U": U}

    def of(self, axis):
        b, a = {"F": ("S", "U"), "S": ("F", "U"), "U": ("S", "F")}[axis]
        return self.axes[axis], self.axes[b], self.axes[a]


def surface(f, b0, b1, a0, a1, step=0.05):
    """Dense points of the wall F = f over b (S) and a (U)."""
    bb, aa = np.meshgrid(np.arange(b0, b1, step), np.arange(a0, a1, step))
    return np.c_[np.full(bb.size, f), bb.ravel(), aa.ravel()]


def makeLargePlanes(dense):
    lp = LargePlanes.__new__(LargePlanes)
    lp.frame, lp.dense, lp.log = AxisFrame(), dense, lambda s: None
    return lp


def wall(axis, off, rect, src="photo", cuts=()):
    return dict(axis=axis, off=off, rect=list(rect), dom="exterior", src=src, yaw=0.0, support=20.0, cuts=list(cuts))


def test_ghost_wall_in_foliage_is_removed():
    # facade F = 0 over b in [-6, 5], measured only on [-4, 5]; a foliage "wall" at F = 2 over [-7, -1] with no dense
    # surface on it, mostly seen through
    dense = surface(0.0, -6.0, 5.0, 0.0, 6.0)
    lp = makeLargePlanes(dense)
    facade = wall("F", 0.0, (-4.0, 5.0, 0.0, 6.0))
    ghost = wall("F", 2.0, (-7.0, -1.0, 0.0, 6.0), src="measured", cuts=[(-6.0, -1.5, 0.5, 5.5)])
    kept = lp._removeGhosts([facade, ghost])
    assert kept == [facade]


def test_unverified_wall_with_dense_surface_is_kept():
    dense = surface(2.0, -7.0, -1.0, 0.0, 6.0)
    lp = makeLargePlanes(dense)
    real = wall("F", 2.0, (-7.0, -1.0, 0.0, 6.0), src="measured")
    assert lp._removeGhosts([real]) == [real]


def test_extension_stops_at_the_corner():
    # facade F = 0 really spans b in [-6, 5]; a side wall S = 5 crosses it (F from -1 to 3)
    dense = surface(0.0, -6.0, 6.5, 0.0, 6.0)
    lp = makeLargePlanes(dense)
    facade = wall("F", 0.0, (-4.0, 4.6, 0.0, 6.0))
    side = wall("S", 5.0, (-1.0, 3.0, 0.0, 6.0))
    lp._extendAlongDense([facade, side])
    assert abs(facade["rect"][0] - (-6.0)) < 0.51          # extended left along the dense surface
    assert facade["rect"][1] == 5.0                         # stopped at the side wall, not beyond


class WalkScene:
    """Camera walking at height 1.6 over a garden (U = 0, F in [0, 6]) then a terrace (U = 0.8, F in [6, 10])."""

    def __init__(self):
        self.frames = list(range(50))
        f = np.linspace(0.5, 9.5, 50)
        self.path = np.c_[f, np.full(50, 2.0), np.where(f < 6, 1.6, 2.4)]

    def centers(self, frames=None):
        return self.path if frames is None else self.path[list(frames)]

    def reliablePoints(self):
        rng = np.random.default_rng(0)
        g = np.c_[rng.uniform(0, 6, 20000), rng.uniform(-1, 5, 20000), rng.normal(0, 0.01, 20000)]
        t = np.c_[rng.uniform(6, 10, 15000), rng.uniform(-1, 5, 15000), 0.8 + rng.normal(0, 0.01, 15000)]
        return np.r_[g, t]


def test_floors_under_the_camera_path():
    lp = makeLargePlanes(None)
    lp.scene, lp.domains = WalkScene(), [("ext", 0, 49, False)]
    terrace = dict(axis="U", off=0.7, rect=[0.0, 4.0, 6.5, 9.5], dom="exterior", src="stairs", name="terrace", cuts=[])
    planes = [terrace]
    lp._floors(planes)
    floors = [p for p in planes if p.get("src") == "cameras"]
    heights = sorted({round(p["off"], 1) for p in floors})
    assert heights == [0.0, 0.8], heights
    assert abs(terrace["off"] - 0.8) < 0.05                  # rule-based level at the measured height
    garden = [p for p in floors if abs(p["off"]) < 0.1]
    area = sum((p["rect"][1] - p["rect"][0]) * (p["rect"][3] - p["rect"][2]) for p in garden)
    assert area > 0.6 * 6 * 6                                # garden mostly covered (within 4 u of the path)
