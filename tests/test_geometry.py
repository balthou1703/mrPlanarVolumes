"""Geometric helpers: rectangles on grids, rectangle difference, periodic comb, OBJ writing."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from planarVolumes import geometry  # noqa: E402


def test_maxRect_finds_largest_block():
    m = np.zeros((10, 12), bool)
    m[2:6, 3:10] = True          # 4 x 7 = 28
    m[7:9, 0:3] = True           # 2 x 3
    area, x0, y0, x1, y1 = geometry.maxRect(m)
    assert (area, x0, y0, x1, y1) == (28, 3, 2, 10, 6)


def test_maxRect_empty():
    assert geometry.maxRect(np.zeros((3, 3), bool))[0] == 0


def test_greedyRects_covers_l_shape():
    m = np.zeros((10, 10), bool)
    m[0:10, 0:3] = True
    m[7:10, 0:10] = True
    rects = geometry.greedyRects(m)
    covered = np.zeros_like(m)
    for r0, r1, c0, c1 in rects:
        covered[r0:r1, c0:c1] = True
    assert (covered == m).all()


def test_rectSubtract_hole_in_the_middle():
    parts = geometry.rectSubtract((0, 10, 0, 5), (4, 6, 1, 3))
    area = sum((b1 - b0) * (a1 - a0) for b0, b1, a0, a1 in parts)
    assert np.isclose(area, 50 - 4)
    assert geometry.rectSubtract((0, 1, 0, 1), (2, 3, 2, 3)) == [(0, 1, 0, 1)]


def test_rectGap_and_overlap():
    assert geometry.rectGap((0, 1, 0, 1), (3, 4, 0, 1)) == 2
    assert geometry.rectGap((0, 2, 0, 2), (1, 3, 1, 3)) == 0
    assert geometry.rectsOverlap((0, 2, 0, 2), (1, 3, 1, 3))
    assert not geometry.rectsOverlap((0, 1, 0, 1), (1, 2, 0, 1))


def test_comb_recovers_step_and_phase():
    rng = np.random.default_rng(0)
    step, phase = 0.17, 0.05
    heights = phase + step * rng.integers(0, 10, 2000) + rng.normal(0, 0.005, 2000)
    r, s, ph = geometry.comb(heights, np.ones(len(heights)), 0.12, 0.25)
    assert r > 0.9
    assert abs(s - step) < 0.003
    assert abs(ph - phase) < 0.01


def test_robustDepthScale_ignores_outliers():
    rng = np.random.default_rng(1)
    mono = rng.uniform(1, 10, 500)
    track = 2.0 * mono + 0.5
    track[:50] *= 3                                       # foreground / wrong matches
    a, b = geometry.robustDepthScale(mono, track)
    assert abs(a - 2.0) < 0.05 and abs(b - 0.5) < 0.2


def test_planeLabel():
    assert geometry.planeLabel("F", -3.96) == "Fm4p0"
    assert geometry.planeLabel("U", 57.1) == "U57p1"


def test_writeObj_meshroom_frame(tmp_path):
    path = str(tmp_path / "a.obj")
    quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    n = geometry.writeObj(path, [("wall", [quad]), ("floor", [quad + 1])], flip=geometry.MESHROOM_FLIP, scale=0.5)
    assert n == 8
    V = geometry.readObjVertices(path)
    assert np.allclose(V[2], [0.5, -0.5, 0])
    assert open(path).read().count("o ") == 2
