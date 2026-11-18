"""Global planes from the per-frame segments of a domain.

- plane distances from the segments anchored on the track only (monocular depth alone drifts by several %);
- monocular-only segments attached to the nearest anchored plane (same family, gap < TOL_MOGE x depth);
- the remaining monocular-only segments grouped into "moge" candidate planes (validated later by the photometric
  refinement, which removes them when it fails);
- extent: 2D occupancy -> successive maximal rectangles.
"""
import os
import pickle

import cv2
import numpy as np

from .families import upFamily
from .geometry import maxRect, writeObj

PLANES_FILE = "planes.pkl"
CELL = 0.08
TOL_TRK = 0.012          # grouping of anchored segments (fraction of depth)
TOL_MOGE = 0.06          # attachment of monocular-only segments
MIN_FRAMES = 3
MIN_RECT = 0.25          # smallest rectangle (u2)
MOGE_MIN_FRAMES, MOGE_MIN_AREA = 4, 2500     # candidates without track: seen in >= 4 frames, large enough


def _groupBy(offsets, depths, weights, tol, useDepthTol):
    """Greedy 1D grouping of offsets around the heaviest ones -> list of index arrays."""
    used = np.zeros(len(offsets), bool)
    groups = []
    for i in np.argsort(-weights):
        if used[i]:
            continue
        width = tol * depths[i] + (0.02 if useDepthTol else 0.0)
        g = (~used) & (np.abs(offsets - offsets[i]) < width)
        o = np.average(offsets[g], weights=weights[g])
        g = (~used) & (np.abs(offsets - o) < width)
        used |= g
        groups.append(np.where(g)[0])
    return groups


def clusterPlanes(scene, segments, families, domainDir, frameMin, frameMax, log=print):
    """Planes of a domain (frames in [frameMin, frameMax]) -> <domainDir>/planes.pkl, planes.obj."""
    segs = [dict(s) for s in segments if frameMin <= s["frame"] <= frameMax]
    up = upFamily(families, scene)
    planes = []
    for f in range(len(families)):
        S = [s for s in segs if s["fam"] == f]
        T = [s for s in S if s["src"] == "trk"]
        off = np.array([s["off"] for s in T])
        dep = np.array([s["depth"] for s in T])
        wgt = np.array([s["area"] * s["ntrk"] for s in T], float)
        clusters = []
        for g in (_groupBy(off, dep, wgt, TOL_TRK, True) if len(T) else []):
            clusters.append(dict(off=np.average(off[g], weights=wgt[g]), members=[T[j] for j in g]))
        if clusters:                                   # attach monocular-only segments
            co = np.array([c["off"] for c in clusters])
            for s in S:
                if s["src"] == "trk":
                    continue
                dd = np.abs(co - s["off"])
                j = int(np.argmin(dd))
                if dd[j] < TOL_MOGE * s["depth"]:
                    clusters[j]["members"].append(s)
                    s["_attached"] = True
        rest = [s for s in S if s["src"] != "trk" and not s.get("_attached")]
        if rest:                                       # loose grouping (monocular scale drift ~6 %)
            offR = np.array([s["off"] for s in rest])
            depR = np.array([s["depth"] for s in rest])
            wR = np.array([s["area"] for s in rest], float)
            for g in _groupBy(offR, depR, wR, TOL_MOGE, False):
                members = [rest[j] for j in g]
                if len({s["frame"] for s in members}) >= MOGE_MIN_FRAMES and wR[g].sum() >= MOGE_MIN_AREA:
                    clusters.append(dict(off=np.average(offR[g], weights=wR[g]), members=members))
        for c in clusters:
            frames = sorted({s["frame"] for s in c["members"]})
            if len(frames) < MIN_FRAMES:
                continue
            pts = np.concatenate([s["pts"] for s in c["members"]])
            pts = pts + (c["off"] - pts @ families[f])[:, None] * families[f]
            planes.append(dict(fam=f, n=families[f], off=c["off"], frames=frames, pts=pts,
                               ntrk=sum(s["src"] == "trk" for s in c["members"]), nseg=len(c["members"]),
                               src="trk" if any(s["src"] == "trk" for s in c["members"]) else "moge"))
    quads = []
    for pi, p in enumerate(planes):
        n = p["n"]
        a = up - (up @ n) * n
        if np.linalg.norm(a) < 0.3:
            a = families[np.argmin(np.abs(families @ up))]
            a = a - (a @ n) * n
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        uv = np.stack([p["pts"] @ b, p["pts"] @ a], 1)
        lo, hi = np.percentile(uv, [0.2, 99.8], 0)
        uv = uv[np.all((uv >= lo) & (uv <= hi), 1)]
        p.update(a=a, b=b, lo=lo, hi=hi)
        sz = np.ceil((hi - lo) / CELL).astype(int) + 1
        cell = CELL * np.sqrt(sz.prod() / 1.5e5) if sz.prod() > 1.5e5 else CELL   # coarser grid for huge planes
        sz = np.ceil((hi - lo) / cell).astype(int) + 1
        occ = np.zeros((sz[1], sz[0]), np.uint16)
        q = ((uv - lo) / cell).astype(int)
        np.add.at(occ, (q[:, 1], q[:, 0]), 1)
        m = (occ >= 2).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        for _ in range(12):
            area, x0, y0, x1, y1 = maxRect(m)
            if area * cell ** 2 < MIN_RECT:
                break
            m[y0:y1, x0:x1] = 0
            u0, v0 = lo + np.array([x0, y0]) * cell
            u1, v1 = lo + np.array([x1, y1]) * cell
            corners = [u * b + v * a + p["off"] * n for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
            quads.append(dict(V=np.array(corners), fam=p["fam"], plane=pi, off=p["off"], area=area * cell ** 2,
                              frames=p["frames"], ntrk=p["ntrk"]))
    log("%s: %d planes (%d monocular-only candidates), %d rectangles" % (
        os.path.basename(domainDir), len(planes), sum(p["src"] == "moge" for p in planes), len(quads)))
    out = os.path.join(domainDir, PLANES_FILE)
    with open(out, "wb") as f:
        pickle.dump(dict(planes=[{k: v for k, v in p.items() if k != "pts"} for p in planes], quads=quads, up=up), f)
    writeObj(out.replace(".pkl", ".obj"), [("plane%03d_q%03d_fam%02d" % (q["plane"], i, q["fam"]), [q["V"]])
                                           for i, q in enumerate(quads)])
    return planes
