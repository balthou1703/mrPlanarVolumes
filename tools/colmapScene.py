"""Development tool: build a scene file (scene.npz) from a binary COLMAP model (PINHOLE cameras), e.g. to compare the
pipeline with a previous run that used a COLMAP model.

Usage: python tools/colmapScene.py <sparse folder> <images folder> <scene.npz> [scale]
The model is taken as already in the working frame; scale (default: <sparse>/scale.txt, else 1) is only recorded to
write the Meshroom frame outputs.
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from planarVolumes.scene import Scene  # noqa: E402

NB_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def qvecToRotation(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def loadColmap(sparse, imagesFolder, scale=None):
    cams = {}
    with open(os.path.join(sparse, "cameras.bin"), "rb") as f:
        n, = _read(f, "Q")
        for _ in range(n):
            cid, model, w, h = _read(f, "iiQQ")
            p = _read(f, "d" * NB_PARAMS[model])
            cams[cid] = ((p[0], p[0], p[1], p[2]) if model == 0 else tuple(p[:4]), (w, h))
    with open(os.path.join(sparse, "points3D.bin"), "rb") as f:
        n, = _read(f, "Q")
        ids, xyz, rgb, tracks = [], [], [], []
        for _ in range(n):
            pid, x, y, z, r, g, b, _e = _read(f, "QdddBBBd")
            tl, = _read(f, "Q")
            f.seek(8 * tl, 1)
            ids.append(pid)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
            tracks.append(tl)
    index = {pid: i for i, pid in enumerate(ids)}
    names, K, sizes, R, t, obs = [], [], [], [], [], []
    with open(os.path.join(sparse, "images.bin"), "rb") as f:
        n, = _read(f, "Q")
        for _ in range(n):
            _iid, = _read(f, "i")
            q = np.array(_read(f, "dddd"))
            tt = np.array(_read(f, "ddd"))
            cid, = _read(f, "i")
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            n2d, = _read(f, "Q")
            data = np.frombuffer(f.read(24 * n2d), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            sel = data["id"] >= 0
            pids = np.array([index.get(int(i), -1) for i in data["id"][sel]], np.int64)
            ok = pids >= 0
            names.append(name.decode())
            K.append(cams[cid][0])
            sizes.append(cams[cid][1])
            R.append(qvecToRotation(q))
            t.append(tt)
            obs.append((np.stack([data["x"][sel][ok], data["y"][sel][ok]], 1), pids[ok]))
    if scale is None:
        fs = os.path.join(sparse, "scale.txt")
        scale = float(open(fs).read().split()[0]) if os.path.isfile(fs) else 1.0
    return Scene(names, K, sizes, R, t, xyz, rgb, tracks, obs, imagesFolder, scale)


if __name__ == "__main__":
    s = loadColmap(sys.argv[1], sys.argv[2], float(sys.argv[4]) if len(sys.argv) > 4 else None)
    s.save(sys.argv[3])
    print("%d views, %d points, scale %.6g -> %s" % (len(s.frames), len(s.points), s.scale, sys.argv[3]))
