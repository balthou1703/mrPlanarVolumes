"""Development tool: hidden-line wireframe of a PlanarVolumes output over the plates, for visual checks.

Usage: python tools/overlayWire.py <scene.npz> <volumes_work.obj> <output folder> <prefix> frame [frame ...]
-> <prefix>_<frame>.jpg (plate | plate + wireframe) and <prefix>_sheet.jpg. The OBJ is the working frame output.
Colors: white walls, yellow reveals, cyan glazing, orange stairs, green floors, magenta dense mesh planes.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from planarVolumes.scene import Scene  # noqa: E402

COLORS = [("stairs", (60, 140, 255)), ("reveal", (60, 220, 255)), ("glazing", (255, 220, 80)), ("floor", (90, 230, 90)),
          ("ground", (90, 230, 90)), ("terrace", (90, 230, 90)), ("dense_", (255, 120, 255))]
NEAR = 0.05


def readPolygons(path):
    V, polys, cols = [], [], []
    col = (255, 255, 255)
    for line in open(path):
        if line.startswith("v "):
            V.append(line.split()[1:4])
        elif line.startswith(("o ", "g ")):
            name = line.split()[1]
            col = next((c for k, c in COLORS if k in name), (255, 255, 255))
        elif line.startswith("f "):
            polys.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
            cols.append(col)
    return np.array(V, float), polys, cols


def clipNear(P):
    """Camera space polygon clipped by the near plane z = NEAR."""
    out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        ia, ib = a[2] >= NEAR, b[2] >= NEAR
        if ia:
            out.append(a)
        if ia != ib:
            out.append(a + (NEAR - a[2]) / (b[2] - a[2]) * (b - a))
    return np.array(out)


def render(scene, V, polys, cols, frame):
    i = scene.index(frame)
    fx, fy, cx, cy = scene.K[i]

    def proj(P):
        return np.stack([fx * P[:, 0] / P[:, 2] + cx, fy * P[:, 1] / P[:, 2] + cy], 1)
    plate = scene.image(frame)
    H, W = plate.shape[:2]
    Xc = V @ scene.R[i].T + scene.t[i]
    cp = [(k, clipNear(Xc[p])) for k, p in enumerate(polys)]
    cp = [(k, P) for k, P in cp if len(P) >= 3]
    ss = 2                                               # id buffer (painter, far to near) for hidden lines
    ids = np.full((H * ss, W * ss), -1, np.int32)
    for k, P in sorted(cp, key=lambda kp: -kp[1][:, 2].mean()):
        uv = np.clip(proj(P) * ss, -1e5, 1e5)
        cv2.fillPoly(ids, [np.round(uv * 4).astype(np.int32)], int(k), shift=2)
    idMax = cv2.dilate(ids.astype(np.float32), np.ones((5, 5), np.uint8))
    idMin = -cv2.dilate(-ids.astype(np.float32), np.ones((5, 5), np.uint8))
    wire = (plate * 0.55).astype(np.uint8)
    for k, P in cp:
        for j in range(len(P)):
            a, b = P[j], P[(j + 1) % len(P)]
            uv = proj(a + np.linspace(0, 1, 60)[:, None] * (b - a)) * ss
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < W * ss) & (uv[:, 1] >= 0) & (uv[:, 1] < H * ss)
            if not ok.any():
                continue
            u, v = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
            vis = (ids[v, u] == k) | (idMax[v, u] == k) | (idMin[v, u] == k)
            seg = uv[ok] / ss
            for m in range(len(seg) - 1):
                if vis[m] and vis[m + 1]:
                    cv2.line(wire, tuple(np.round(seg[m] * 8).astype(int)), tuple(np.round(seg[m + 1] * 8).astype(int)),
                             cols[k], 2, cv2.LINE_AA, shift=3)
    for img, text in ((plate, "plate %d" % frame), (wire, "PlanarVolumes %d" % frame)):
        cv2.putText(img, text, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, text, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack([plate, wire])


def main():
    scene = Scene.load(sys.argv[1])
    V, polys, cols = readPolygons(sys.argv[2])
    out, prefix = sys.argv[3], sys.argv[4]
    os.makedirs(out, exist_ok=True)
    tiles = []
    for frame in map(int, sys.argv[5:]):
        if not scene.hasFrame(frame):
            print("frame %d not in the scene" % frame)
            continue
        pair = render(scene, V, polys, cols, frame)
        cv2.imwrite(os.path.join(out, "%s_%04d.jpg" % (prefix, frame)), pair, [cv2.IMWRITE_JPEG_QUALITY, 90])
        tiles.append(cv2.resize(pair, (pair.shape[1] // 3, pair.shape[0] // 3)))
    rows = [np.hstack(tiles[i:i + 2]) if i + 1 < len(tiles) else np.hstack([tiles[i], np.zeros_like(tiles[i])])
            for i in range(0, len(tiles), 2)]
    cv2.imwrite(os.path.join(out, "%s_sheet.jpg" % prefix), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("->", out)


if __name__ == "__main__":
    main()
