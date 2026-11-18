"""Scene: frame numbers, projection, track observations, save / load."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from planarVolumes.scene import Scene, frameNumber  # noqa: E402


def makeScene():
    names = ["shot.0012.jpg", "shot.0010.jpg", "shot.0011.jpg"]       # unsorted on purpose
    K = [(1000, 1000, 960, 540)] * 3
    R = [np.eye(3)] * 3
    t = [np.array([0, 0, 5.0 + i]) for i in range(3)]
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    obs = []
    for ti in t:
        Xc = points + ti
        xy = np.stack([1000 * Xc[:, 0] / Xc[:, 2] + 960, 1000 * Xc[:, 1] / Xc[:, 2] + 540], 1)
        obs.append((xy, np.arange(3)))
    return Scene(names, K, [(1920, 1080)] * 3, R, t, points, np.zeros((3, 3)), [3, 3, 2], obs, "", 2.0)


def test_frameNumber():
    assert frameNumber("A001_C010.0953.exr") == 953
    assert frameNumber("img_12.jpg") == 12
    assert frameNumber("nodigits.png") == -1


def test_views_sorted_by_frame():
    s = makeScene()
    assert s.frames == [10, 11, 12]
    assert np.allclose(s.center(12), [0, 0, -5.0])            # first pose of the input order (shot.0012)


def test_project_and_track_observations():
    s = makeScene()
    uv, z = s.project(10, np.array([[1.0, 0, 0]]))
    assert np.allclose(uv, [[1000 / 6 + 960, 540]]) and np.allclose(z, 6.0)
    xy, X = s.trackObservations(11)
    assert len(X) == 2                                          # the point seen twice is not reliable


def test_save_load_roundtrip(tmp_path):
    s = makeScene()
    path = str(tmp_path / "scene.npz")
    s.save(path)
    r = Scene.load(path)
    assert r.frames == s.frames and r.scale == 2.0
    assert np.allclose(r.t, s.t) and np.allclose(r.points, s.points)
    for (a, ia), (b, ib) in zip(r.observations, s.observations):
        assert np.allclose(a, b) and (ia == ib).all()
