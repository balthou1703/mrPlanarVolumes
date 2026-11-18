"""Large planes: one slab per surface (wall, floor), aligned on the dominant frame (up, facade normal, side), then
refined photometrically over its whole extent.

1. cells of all the refined planes of the domains -> weighted cloud (area, validation);
2. per axis (facade F, side S, up U): histogram of the distances of the cells oriented like the axis (<= 35 deg)
   -> peaks = planes; close layers (< 0.8 u: shutters, monocular smears) are absorbed;
3. connected components of the footprint in the plane -> rectangles (percentiles 1-99);
4. photometric plane sweep of each large plane over its real footprint: distance + yaw;
5. ghost walls (foliage) removed: an unverified outdoor wall mostly seen through, or without dense surface;
   walls extended along the dense surface up to the corners;
6. openings: seen-through votes of the carving reported on each wall;
7. horizontal levels: stairs foot (ground) + landing (terrace, between the facade and the retaining wall, notched
   by the stairs), then the floors under the camera path (height map of the lowest points, one slab per level);
8. joints: bottom of the walls lowered to the lowest touching level, floor edges pushed to the walls.
"""
import os
import pickle

import cv2
import numpy as np

from .geometry import MESHROOM_FLIP, planeLabel, readObjVertices, rectGap, rectSubtract, writeObj
from .stairs import loadStairs

LARGE_PLANES_FILE = "largePlanes.pkl"
TOL = 0.6            # half thickness of a large plane (u)
MERGE = 0.8          # closer peaks -> same plane
MIN_SUPPORT = 2.5    # smallest measured area (u2)
COS35 = np.cos(np.radians(35))
# ghost walls (foliage) and extension along the dense surface
SEE_THROUGH = 0.4    # seen-through fraction of the footprint above which an unverified wall is dropped
DENSE_MIN = 0.15     # smallest dense coverage (occupied cells / footprint) of an unverified wall
DENSE_BAND = 0.2     # half thickness of the band around a plane (u)
DENSE_COS = np.cos(np.radians(30))   # dense surface points oriented like the plane
DENSE_PLANARITY = 0.1                # smallest / middle PCA eigenvalue of a planar neighbourhood
EXTEND_MIN = 0.5     # smallest dense coverage of a wall to extend it
EXT_STEP, EXT_MAX = 0.5, 2.0   # extension step and maximum length (u)
# floors under the camera path
FLOOR_CELL = 0.25           # height map cell (u)
FLOOR_RADIUS = 4.0          # points within this horizontal distance of the camera path (u)
FLOOR_BELOW_CAMERA = 0.3    # floor points are at least this much below the camera (u)
FLOOR_MIN_CAMERAS = 0.05    # smallest level: fraction of the cameras of the domain standing above it
FLOOR_WALL_GAP = 1.5        # a wall whose measured bottom is up to this much above a floor still bounds it (u)
FLOOR_MIN_EYE = 0.6         # a level with the camera less than this fraction of its height above the main floor: furniture
FLOOR_MIN_AREA = 1.0        # smallest floor rectangle (u2)
FLOOR_MIN_WIDTH = 1.0       # thinnest floor rectangle (u)
FLOOR_STAIRS_MARGIN = 0.3   # the stairs footprint (+ margin, u) is not a floor
INPLANE = {"F": ("S", "U"), "S": ("F", "U"), "U": ("S", "F")}     # (horizontal b, a) axes of each plane axis


class Frame:
    """Dominant frame of the scene: F (main facade normal), S (side), U (up)."""

    def __init__(self, up, families):
        up = up / np.linalg.norm(up)
        f0 = next(f for f in families if abs(f @ up) < 0.3)       # most important wall family = facade normal
        f0 = f0 - (f0 @ up) * up
        f0 /= np.linalg.norm(f0)
        self.axes = {"F": f0, "S": np.cross(up, f0), "U": up}

    def of(self, axis):
        """(normal, b, a) unit vectors of a plane of the given axis."""
        b, a = INPLANE[axis]
        return self.axes[axis], self.axes[b], self.axes[a]


def planarPoints(points, voxel=0.05, k=16):
    """Locally planar points of a dense surface and their normals (voxel subsampling, PCA of the k nearest points);
    foliage and clutter are not planar.

    Returns:
        tuple: points (m, 3), unit normals (m, 3).
    """
    from scipy.spatial import cKDTree
    key = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    P = points[np.sort(first)]
    _, nn = cKDTree(P).query(P, k=min(k, len(P)), workers=-1)
    Q = P[nn] - P[nn].mean(1, keepdims=True)
    w, vec = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))
    planar = w[:, 0] < DENSE_PLANARITY * w[:, 1]
    return P[planar], vec[planar, :, 0]


def _overlapArea(r1, r2):
    return max(0.0, min(r1[1], r2[1]) - max(r1[0], r2[0])) * max(0.0, min(r1[3], r2[3]) - max(r1[2], r2[2]))


def _clipToWalls(rect, walls, level, cameras):
    """Cut a floor rectangle (S range, F range) at the walls it crosses, keeping the side of most of its cameras
    ((F, S) positions). A wall crosses the floor when its plane splits the rectangle and its extent covers the floor
    level (its bottom is lowered to the floors only later: up to FLOOR_WALL_GAP above) and overlaps the rectangle
    sideways."""
    for w in walls:
        if w["rect"][2] - FLOOR_WALL_GAP > level or w["rect"][3] < level + 0.5:
            continue
        if w["axis"] == "F":                       # wall F = off: cuts the F range (a: indices 2, 3), spans S (b)
            lo_, hi_, side, cam = 2, 3, (w["rect"][0], w["rect"][1]), cameras[:, 0] if len(cameras) else cameras
            other = (rect[0], rect[1])
        else:                                      # wall S = off: cuts the S range (indices 0, 1), spans F
            lo_, hi_, side, cam = 0, 1, (w["rect"][0], w["rect"][1]), cameras[:, 1] if len(cameras) else cameras
            other = (rect[2], rect[3])
        off = w["off"]
        if not rect[lo_] + 0.3 < off < rect[hi_] - 0.3 or min(other[1], side[1]) - max(other[0], side[0]) < 0.5:
            continue
        if len(cam) and (cam > off).mean() > 0.5:
            rect[lo_] = off
        else:
            rect[hi_] = off


def _cellsOfPlane(g):
    p = g["p"]
    ys, xs = np.nonzero(g["mask"])
    u = g["lo"][0] + (xs + .5) * g["cell"]
    v = g["lo"][1] + (ys + .5) * g["cell"]
    height = p["off"] + g["delta"] + g["tx"] * (u - g["uc"][0]) + g["ty"] * (v - g["uc"][1])
    X = u[:, None] * p["b"] + v[:, None] * p["a"] + height[:, None] * p["n"]
    n = p["n"] - g["tx"] * p["b"] - g["ty"] * p["a"]
    return X, n / np.linalg.norm(n)


class LargePlanes:
    """Large planes of a scene from the finalized planes of its domains.

    Args:
        scene: Scene.
        work: working folder.
        domains: list of (name, frameMin, frameMax, interior).
        refiner: Refiner (photometric plane sweep).
        densePoints: optional (n, 3) vertices of the dense mesh in the working frame.
    """

    def __init__(self, scene, work, domains, refiner, densePoints=None, log=print):
        self.scene, self.work, self.domains, self.refiner, self.log = scene, work, domains, refiner, log
        self.dense = densePoints
        self.denseNormals = None
        if densePoints is not None:                      # wall surface = locally planar dense points (not foliage)
            self.densePlanar, self.denseNormals = planarPoints(densePoints)
        self.sources = []
        for k, (name, fmin, fmax, interior) in enumerate(domains):
            with open(os.path.join(work, name, "final.pkl"), "rb") as f:
                self.sources.append(("interior" if interior else "exterior", k, pickle.load(f)))
        self.firstInterior = min([fmin for _, fmin, _, interior in domains if interior], default=None)
        up = self.sources[0][2]["up"]
        self.frame = Frame(up, np.load(os.path.join(work, "families.npy")))

    # --- 1-3. cells -> peaks -> rectangles
    def _cloud(self):
        X, N, W, interior, pid, framesOf = [], [], [], [], [], {}
        for dom, k, src in self.sources:
            for g in src["planes"]:
                framesOf[g["i"] + 1000 * k] = g["p"]["frames"]
                if g["mask"].sum() == 0:
                    continue
                if dom == "exterior" and self.firstInterior is not None and min(g["p"]["frames"]) >= self.firstInterior - 200:
                    continue                             # seen only from inside: duplicates of the interior domain
                x, n = _cellsOfPlane(g)
                X.append(x)
                N.append(np.repeat(n[None], len(x), 0))
                W.append(np.full(len(x), g["cell"] ** 2 * (1.0 if g["ok"] else 0.6)))
                interior.append(np.full(len(x), dom == "interior"))
                pid.append(np.full(len(x), g["i"] + 1000 * k))
        self.X, self.N, self.W = np.concatenate(X), np.concatenate(N), np.concatenate(W)
        self.interior, self.pid, self.framesOf = np.concatenate(interior), np.concatenate(pid), framesOf
        self.log("large planes: %d cells, %.0f u2" % (len(self.X), self.W.sum()))

    def _peaks(self, axis):
        A = self.frame.axes[axis]
        ok = np.abs(self.N @ A) > COS35
        d, w = self.X[ok] @ A, self.W[ok]
        lo, hi = np.floor(d.min()) - 1, np.ceil(d.max()) + 1
        H, e = np.histogram(d, bins=np.arange(lo, hi, 0.05), weights=w)
        Hs = cv2.GaussianBlur(H[None].astype(np.float32), (0, 0), 3).ravel()
        out = []
        for i in np.argsort(-Hs):
            if Hs[i] * 3 * np.sqrt(2 * np.pi) < MIN_SUPPORT * 0.5:       # ~ area of the peak
                break
            c = e[i] + 0.025
            if all(abs(c - o) > MERGE for o in out):
                out.append(c)
        return out

    def _measure(self):
        planes = []
        used = np.zeros(len(self.X), bool)           # exclusive: one cell -> one large plane
        for axis in ("F", "S", "U"):
            A, B, Av = self.frame.of(axis)
            for c in self._peaks(axis):
                inl = (np.abs(self.N @ A) > COS35) & (np.abs(self.X @ A - c) < MERGE) & ~used
                if self.W[inl].sum() < MIN_SUPPORT:
                    continue
                idx = np.nonzero(inl)[0]
                o = np.argsort(self.X[idx] @ A)
                cw = np.cumsum(self.W[idx][o])
                off = float((self.X[idx] @ A)[o][np.searchsorted(cw, cw[-1] / 2)])
                used[idx] = True
                ub, ua = self.X[idx] @ B, self.X[idx] @ Av
                G = 0.25
                ib = ((ub - ub.min()) / G).astype(int)
                ia = ((ua - ua.min()) / G).astype(int)
                occ = np.zeros((ia.max() + 1, ib.max() + 1), np.uint8)
                occ[ia, ib] = 1
                nlab, lab = cv2.connectedComponents(cv2.dilate(occ, np.ones((9, 9), np.uint8)))
                for L in range(1, nlab):
                    k = lab[ia, ib] == L
                    support = self.W[idx][k].sum()
                    if support < MIN_SUPPORT:
                        continue
                    b0, b1 = np.percentile(ub[k], [1, 99])
                    a0, a1 = np.percentile(ua[k], [1, 99])
                    interior = self.interior[idx][k].mean() > 0.5
                    pids = np.unique(self.pid[idx][k])
                    frames = sorted({f for p_ in pids for f in self.framesOf[p_]})
                    planes.append(dict(axis=axis, off=off, rect=[b0, b1, a0, a1], support=float(support),
                                       dom="interior" if interior else "exterior", pts=self.X[idx][k], frames=frames,
                                       yaw=0.0, src="measured"))
                    self.log("  plane %s d=%7.2f %s rect b %6.2f..%6.2f a %6.2f..%6.2f measured %5.1f u2" % (
                        axis, off, planes[-1]["dom"], b0, b1, a0, a1, support))
        return planes

    @staticmethod
    def _fuse(planes):
        """Merge the pieces of a same wall (same axis, distances < MERGE, neighbouring footprints)."""
        fused = True
        while fused:
            fused = False
            for i in range(len(planes)):
                for j in range(i + 1, len(planes)):
                    p, q = planes[i], planes[j]
                    if p["axis"] != q["axis"] or p["dom"] != q["dom"] or abs(p["off"] - q["off"]) > MERGE \
                            or rectGap(p["rect"], q["rect"]) > 2.0:
                        continue
                    w = p["support"] + q["support"]
                    p["off"] = (p["off"] * p["support"] + q["off"] * q["support"]) / w
                    p["rect"] = [min(p["rect"][0], q["rect"][0]), max(p["rect"][1], q["rect"][1]),
                                 min(p["rect"][2], q["rect"][2]), max(p["rect"][3], q["rect"][3])]
                    p["pts"] = np.concatenate([p["pts"], q["pts"]])
                    p["support"] = w
                    p["frames"] = sorted(set(p["frames"]) | set(q["frames"]))
                    del planes[j]
                    fused = True
                    break
                if fused:
                    break
        return planes

    # --- 4. photometric refinement
    def _refine(self, pl):
        if pl["axis"] == "U":
            return
        A, B, Av = self.frame.of(pl["axis"])
        cell = 0.1
        ub, ua = pl["pts"] @ B, pl["pts"] @ Av
        lo = np.array([ub.min(), ua.min()])
        ix = ((ub - lo[0]) / cell).astype(int)
        iy = ((ua - lo[1]) / cell).astype(int)
        m = np.zeros((iy.max() + 1, ix.max() + 1), np.uint8)
        m[iy, ix] = 1
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        zs = [np.linalg.norm(pl["pts"].mean(0) - self.scene.center(fr))
              for fr in pl["frames"][::max(1, len(pl["frames"]) // 20)] if self.scene.hasFrame(fr)]
        o = dict(plane=dict(n=A, b=B, a=Av, off=pl["off"], frames=pl["frames"], fam=-1),
                 mask=m, lo=lo, cell=cell, zmed=float(np.median(zs)) if zs else 10.0)
        r = self.refiner.refine(o)
        if r is None:
            self.log("  %s %.2f: no views" % (pl["axis"], pl["off"]))
            return
        yaw = np.degrees(np.arctan(r["tx"]))
        gain = r["peak"] - r["flat"]
        ok = r["peak"] > 0.25 and r["contrast"] > 0.04
        pl["pitch"] = float(np.degrees(np.arctan(r["ty"])))
        self.log("  %s %7.2f -> delta %+.3f pitch %+.2f deg yaw %+.2f deg (gain %.3f) NCC %.2f contrast %.2f %s" % (
            pl["axis"], pl["off"], r["delta"], pl["pitch"], yaw, gain, r["peak"], r["contrast"], "OK" if ok else "rejected"))
        if ok:
            pl["off"] += r["delta"]
            pl["src"] = "photo"
            if gain > 0.05 and abs(yaw) < 4.0:           # otherwise strict dominant frame
                pl["yaw"] = float(r["tx"])
                pl["yaw_c"] = float(r["uc"][0])

    # --- 5. ghost walls and extension along the dense surface
    def _onPlane(self, pl):
        """Dense surface points in the band of a plane and oriented like it -> (b, a) coordinates."""
        A, B, Av = self.frame.of(pl["axis"])
        normals = getattr(self, "denseNormals", None)
        P = self.densePlanar if normals is not None else self.dense
        b, a = P @ B, P @ Av
        d = pl["off"] + pl["yaw"] * (b - pl.get("yaw_c", 0.0))
        k = np.abs(P @ A - d) < DENSE_BAND
        if normals is not None:
            k &= np.abs(normals @ A) > DENSE_COS
        return b[k], a[k]

    @staticmethod
    def _coverage(b, a, rect, cell=0.25):
        b0, b1, a0, a1 = rect
        k = (b >= b0) & (b < b1) & (a >= a0) & (a < a1)
        nb, na = max(1, int((b1 - b0) / cell)), max(1, int((a1 - a0) / cell))
        occ = np.zeros((na, nb), bool)
        occ[np.minimum(((a[k] - a0) / cell).astype(int), na - 1), np.minimum(((b[k] - b0) / cell).astype(int), nb - 1)] = True
        return occ.mean()

    @staticmethod
    def _cutFraction(pl):
        b0, b1, a0, a1 = pl["rect"]
        area = max((b1 - b0) * (a1 - a0), 1e-6)
        return sum(max(0, min(c[1], b1) - max(c[0], b0)) * max(0, min(c[3], a1) - max(c[2], a0)) for c in pl["cuts"]) / area

    def _removeGhosts(self, planes):
        """Outdoor walls in foliage: an unverified wall mostly seen through, or (with the dense mesh) any outdoor wall
        without dense surface on its plane: a weak photometric validation on foliage is not enough."""
        keep = []
        for pl in planes:
            if pl["axis"] == "U" or pl["dom"] != "exterior":
                keep.append(pl)
                continue
            unverified = pl["src"] == "measured"
            seen = self._cutFraction(pl)
            cov = self._coverage(*self._onPlane(pl), pl["rect"]) if self.dense is not None else None
            if (unverified and seen > SEE_THROUGH) or (cov is not None and cov < DENSE_MIN):
                self.log("wall %s %.2f dropped (%s; seen through %.0f %%%s)" % (
                    pl["axis"], pl["off"], "unverified" if unverified else "photo", 100 * seen,
                    "" if cov is None else ", dense coverage %.0f %%" % (100 * cov)))
                continue
            keep.append(pl)
        return keep

    def _extendAlongDense(self, planes):
        for pl in planes:
            if pl["axis"] == "U":
                continue
            b, a = self._onPlane(pl)
            a0, a1 = pl["rect"][2], pl["rect"][3]
            ref = self._coverage(b, a, pl["rect"])
            if ref < EXTEND_MIN:
                continue
            # corners: perpendicular walls whose footprint crosses this plane stop the extension
            stops = [w["off"] for w in planes if w["axis"] not in ("U", pl["axis"]) and w["dom"] == pl["dom"]
                     and w["rect"][0] - 0.5 <= pl["off"] <= w["rect"][1] + 0.5
                     and min(a1, w["rect"][3]) - max(a0, w["rect"][2]) > 0.5]
            for side, sgn in ((0, -1), (1, 1)):
                edge = pl["rect"][side]
                ahead = [s for s in stops if (s - edge) * sgn > -0.3]
                end = min(ahead, key=lambda s: abs(s - edge)) if ahead else None
                if end is not None and (end - edge) * sgn <= 0:
                    continue                             # already at the corner
                while abs(edge - pl["rect"][side]) < EXT_MAX:
                    strip = (edge, edge + EXT_STEP) if sgn > 0 else (edge - EXT_STEP, edge)
                    if self._coverage(b, a, (strip[0], strip[1], a0, a1)) < 0.5 * ref:
                        break
                    edge += sgn * EXT_STEP
                    if end is not None and (edge - end) * sgn >= -0.25:
                        edge = end
                        break
                if edge != pl["rect"][side]:
                    self.log("wall %s %.2f extended (%s): b %.2f -> %.2f (dense coverage %.0f %%)" % (
                        pl["axis"], pl["off"], "left" if side == 0 else "right", pl["rect"][side], edge, 100 * ref))
                    pl["rect"][side] = edge

    # --- 6. openings
    def _openings(self, pl, carved, cell=0.1):
        """Seen-through zones of the carving reported on a large wall -> list of rectangles (b0, b1, a0, a1)."""
        A, B, Av = self.frame.of(pl["axis"])
        b0, b1, a0, a1 = pl["rect"]
        nb, na = int((b1 - b0) / cell) + 1, int((a1 - a0) / cell) + 1
        Fs = np.zeros((na, nb), np.float32)
        Ss = np.zeros((na, nb), np.float32)
        for o in carved[pl["dom"]]:
            if o is None or abs(o["plane"]["n"] @ A) < COS35:
                continue
            q = o["plane"]
            ys, xs = np.nonzero((o["F"] > 0) | (o["S"] > 0))
            u = o["lo"][0] + (xs + .5) * o["cell"]
            v = o["lo"][1] + (ys + .5) * o["cell"]
            Xc = u[:, None] * q["b"] + v[:, None] * q["a"] + q["off"] * q["n"]
            k = np.abs(Xc @ A - pl["off"]) < 0.8
            if k.sum() < 50:
                continue
            ib = ((Xc[k] @ B - b0) / cell).astype(int)
            ia = ((Xc[k] @ Av - a0) / cell).astype(int)
            ok = (ib >= 0) & (ia >= 0) & (ib < nb) & (ia < na)
            np.add.at(Fs, (ia[ok], ib[ok]), o["F"][ys[k][ok], xs[k][ok]])
            np.add.at(Ss, (ia[ok], ib[ok]), o["S"][ys[k][ok], xs[k][ok]])
        Fs = cv2.GaussianBlur(Fs, (0, 0), 1.5)
        Ss = cv2.GaussianBlur(Ss, (0, 0), 1.5)
        m = ((Fs >= 2) & (Fs > 1.5 * Ss)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        nlab, lab, st, _ = cv2.connectedComponentsWithStats(m)
        cuts = []
        for L in range(1, nlab):
            x, y, w, h, area = st[L]
            if area * cell ** 2 < 0.8 or h * cell < 1.0 or w * cell < 0.5 or area < 0.5 * w * h:
                continue
            cuts.append([b0 + x * cell, b0 + (x + w) * cell, a0 + y * cell, a0 + (y + h) * cell])
        return cuts

    # --- 7. levels from the stairs (ground at the foot, terrace at the landing)
    def _levels(self, planes, stairs):
        F, S, U = self.frame.axes["F"], self.frame.axes["S"], self.frame.axes["U"]
        wallsF = [p for p in planes if p["axis"] == "F" and p["dom"] == "exterior"]
        if stairs is None or not wallsF:
            return
        Vst = np.concatenate([q["V"] for q in stairs["quads"]])
        hGround, hTerrace = float((Vst @ U).min()), float((Vst @ U).max())
        stRect = [(Vst @ S).min(), (Vst @ S).max(), (Vst @ F).min(), (Vst @ F).max()]
        facade = max(wallsF, key=lambda p: p["support"])
        front = [p for p in wallsF if p["off"] > facade["off"] + 3 and p["rect"][2] < hTerrace]
        self.log("facade d=%.2f; retaining walls %s; ground h=%.2f, terrace h=%.2f" % (
            facade["off"], [round(p["off"], 2) for p in front], hGround, hTerrace))
        if not front:
            return
        retaining = max(front, key=lambda p: p["support"])
        terrace = [(facade["rect"][0], facade["rect"][1], facade["off"], retaining["off"])]
        terrace = [r for t in terrace for r in rectSubtract(t, (stRect[0], stRect[1], stRect[2], stRect[3] + 1))]
        for r in terrace:
            planes.append(dict(axis="U", off=hTerrace, rect=list(r), support=0, dom="exterior", frames=[], yaw=0.0,
                               src="stairs", name="terrace", cuts=[]))
        for p in front:                                   # retaining walls: ground to terrace, notched by the stairs
            p["rect"][2] = min(p["rect"][2], hGround)
            p["rect"][3] = max(p["rect"][3], hTerrace)
            p["cuts"].append([stRect[0], stRect[1], hGround - 1, hTerrace + 1])
        # ground: in front of the retaining wall, extent of the track points at ground level
        P = self.scene.reliablePoints()
        near = (np.abs(P @ U - hGround) < 0.6) & (P @ F > retaining["off"])
        if near.sum() > 50:
            fs = np.percentile(P[near] @ F, [2, 98])
            ss = np.percentile(P[near] @ S, [2, 98])
        else:
            fs = (retaining["off"], retaining["off"] + 15)
            ss = (retaining["rect"][0] - 5, retaining["rect"][1] + 5)
        ss = (max(ss[0], facade["rect"][0] - 6), min(ss[1], facade["rect"][1] + 6))
        planes.append(dict(axis="U", off=hGround, rect=[ss[0], ss[1], retaining["off"], min(fs[1], retaining["off"] + 20)],
                           support=0, dom="exterior", frames=[], yaw=0.0, src="stairs", name="ground", cuts=[]))

    # --- 7b. floors under the cameras
    def _floors(self, planes, stairs=None):
        """Floors where the camera walks: height map of the lowest points (track + dense mesh) around the camera path
        of each domain; the levels are the heights found right under the cameras (furniture is walked around, not
        on); one rectangle per zone of a level connected to the camera positions.

        A rule-based level (terrace, ground from the stairs) at a close height takes the measured height, and the
        measured floor only adds what it does not cover. Outdoor floors seen through a door (same height as an indoor
        floor they overlap) are dropped.
        """
        from scipy.spatial import cKDTree
        F, S, U = self.frame.axes["F"], self.frame.axes["S"], self.frame.axes["U"]
        points = self.scene.reliablePoints()
        if self.dense is not None:
            points = np.r_[self.dense, points]
        q, h = np.c_[points @ F, points @ S], points @ U
        stairsBox = None
        if stairs is not None:
            V = np.concatenate([qd["V"] for qd in stairs["quads"]])
            m = FLOOR_STAIRS_MARGIN
            stairsBox = ((V @ F).min() - m, (V @ F).max() + m, (V @ S).min() - m, (V @ S).max() + m)
        found = []
        for name, fmin, fmax, interior in sorted(self.domains, key=lambda d: not d[3]):     # indoor domains first
            dom = "interior" if interior else "exterior"
            C = self.scene.centers([f for f in self.scene.frames if fmin <= f <= fmax])
            if len(C) < 2:
                continue
            ch, c2 = C @ U, np.c_[C @ F, C @ S]
            d, j = cKDTree(c2).query(q, distance_upper_bound=FLOOR_RADIUS)
            k = np.isfinite(d)
            k[k] &= h[k] < ch[j[k]] - FLOOR_BELOW_CAMERA
            if stairsBox is not None:                    # walking down the stairs: their steps are not floors
                k &= ~((q[:, 0] > stairsBox[0]) & (q[:, 0] < stairsBox[1])
                       & (q[:, 1] > stairsBox[2]) & (q[:, 1] < stairsBox[3]))
            if k.sum() < 100:
                continue
            qk, hk = q[k], h[k]
            lo = qk.min(0)
            n = np.ceil((qk.max(0) - lo) / FLOOR_CELL).astype(int) + 1
            ij = ((qk - lo) / FLOOR_CELL).astype(int)
            key = ij[:, 0] * n[1] + ij[:, 1]
            o = np.lexsort((hk, key))
            keys, cnt = np.unique(key[o], return_counts=True)
            first = np.r_[0, np.cumsum(cnt)[:-1]]
            low = hk[o][first + (cnt * 0.1).astype(int)]       # robust lowest point of the cell
            ok = cnt >= 8
            hm = np.full(n[0] * n[1], np.nan)
            hm[keys[ok]] = low[ok]
            hm = hm.reshape(n)                                  # rows: F, columns: S
            # floor height right under each camera (median of the 3 x 3 cells)
            ci = np.clip(((c2 - lo) / FLOOR_CELL).astype(int), 1, np.maximum(n - 2, 1))
            under = np.array([np.nanmedian(hm[i - 1:i + 2, jj - 1:jj + 2]) if np.isfinite(hm[i - 1:i + 2, jj - 1:jj + 2]).any()
                              else np.nan for i, jj in ci])
            seen = np.isfinite(under)
            if seen.sum() < 5:
                continue
            H, e = np.histogram(under[seen], bins=np.arange(np.nanmin(under) - 0.05, np.nanmax(under) + 0.1, 0.05))
            levels = []
            for i in np.argsort(-H):
                if H[i] < max(5, FLOOR_MIN_CAMERAS * seen.sum()):
                    break
                c = e[i] + 0.025
                if all(abs(c - lv) > 0.15 for lv in levels):
                    levels.append(c)
            # camera height above the floor: a level much closer under the camera than the main one is furniture
            eye = {lv: np.median(ch[seen & (np.abs(under - lv) < 0.1)] - lv) for lv in levels}
            if levels:
                ref = eye[levels[0]]
                for lv in levels[1:]:
                    if eye[lv] < FLOOR_MIN_EYE * ref:
                        self.log("level %s h=%.2f: camera %.2f above it (%.2f on the main floor), furniture: dropped" % (
                            dom, lv, eye[lv], ref))
                levels = [lv for lv in levels if eye[lv] >= FLOOR_MIN_EYE * ref]
            walls = [w for w in planes if w["axis"] != "U" and w["support"] >= 5 and not w.get("name")]
            for level in levels:
                cells = np.isfinite(hm) & (np.abs(hm - level) < 0.1)
                sel = (np.abs(hk - level) < 0.06) & cells.ravel()[key]
                level = float(np.median(hk[sel])) if sel.sum() > 20 else float(level)
                seeds = np.zeros(n, np.uint8)
                at = seen & (np.abs(under - level) < 0.1)
                seeds[ci[at, 0], ci[at, 1]] = 1
                seeds = cv2.dilate(seeds, np.ones((3, 3), np.uint8))
                mask = cv2.morphologyEx(cells.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
                nl, lab = cv2.connectedComponents(mask, connectivity=8)
                for L in np.unique(lab[(seeds > 0) & (mask > 0)]):
                    rows, cols = np.nonzero(lab == L)
                    if len(rows) * FLOOR_CELL ** 2 < FLOOR_MIN_AREA:
                        continue
                    r0, r1 = np.percentile(rows, [2, 98])
                    c0, c1 = np.percentile(cols, [2, 98])
                    rect = [lo[1] + c0 * FLOOR_CELL, lo[1] + (c1 + 1) * FLOOR_CELL,      # b = S (columns)
                            lo[0] + r0 * FLOOR_CELL, lo[0] + (r1 + 1) * FLOOR_CELL]      # a = F (rows)
                    onIt = c2[at]
                    _clipToWalls(rect, walls, level, onIt)
                    area = (rect[1] - rect[0]) * (rect[3] - rect[2])
                    if not interior and any(f["dom"] == "interior" and abs(f["off"] - level) < 0.15
                                            and _overlapArea(f["rect"], rect) > 0.3 * area for f in found):
                        self.log("floor exterior h=%.2f seen through a door: dropped" % level)
                        continue
                    if any(f["dom"] == dom and f["off"] < level - 0.15 and _overlapArea(f["rect"], rect) > 0.5 * area
                           for f in found):
                        self.log("level %s h=%.2f above a floor (furniture, platform): dropped" % (dom, level))
                        continue
                    pieces = [rect]
                    for p in planes:                     # rule-based levels: measured height, no duplicate surface
                        if p["axis"] != "U" or p["dom"] != dom or abs(p["off"] - level) > 0.4:
                            continue
                        if any(rectGap(p["rect"], r) == 0 for r in pieces):
                            if p.get("name"):
                                p["off"] = level
                            pieces = [r for piece in pieces for r in rectSubtract(piece, p["rect"])]
                    for r in pieces:
                        if (r[1] - r[0]) * (r[3] - r[2]) < FLOOR_MIN_AREA or min(r[1] - r[0], r[3] - r[2]) < FLOOR_MIN_WIDTH:
                            continue                     # small pieces and slivers left by the rule-based levels
                        floor = dict(axis="U", off=level, rect=list(r), support=0, dom=dom, frames=[], yaw=0.0,
                                     src="cameras", cuts=[])
                        planes.append(floor)
                        found.append(floor)
                        self.log("floor %s h=%.2f: S %.2f..%.2f F %.2f..%.2f" % (dom, level, *r))

    # --- 8. joints
    @staticmethod
    def _joints(planes):
        levels = [p for p in planes if p["axis"] == "U"]
        walls = [p for p in planes if p["axis"] != "U"]
        for w in walls:
            if w.get("name"):
                continue
            b0, b1, a0, a1 = w["rect"]
            best = None
            for L in levels:                              # floors touching the foot of the wall
                if L["dom"] != w["dom"] and not L.get("name"):
                    continue
                lb0, lb1, la0, la1 = L["rect"]            # U: b = S, a = F
                if w["axis"] == "F":
                    touch = la0 - 1.5 <= w["off"] <= la1 + 1.5 and min(b1, lb1) - max(b0, lb0) > 0.5
                else:
                    touch = lb0 - 1.5 <= w["off"] <= lb1 + 1.5 and min(b1, la1) - max(b0, la0) > 0.5
                if touch and L["off"] < a0 + 0.3 and a0 - L["off"] < 4.5:
                    best = L["off"] if best is None else min(best, L["off"])
            if best is not None:
                w["rect"][2] = best
        for L in levels:
            if L.get("name"):
                continue
            for k, (axis, side) in enumerate((("S", 0), ("S", 1), ("F", 0), ("F", 1))):
                cur = L["rect"][k]
                cands = [w["off"] for w in walls if w["axis"] == axis and w["support"] >= 5
                         and (w["off"] - cur) * (1 if side else -1) > -0.2 and abs(w["off"] - cur) < 2.5
                         and w["rect"][2] - 0.5 <= L["off"] <= w["rect"][3] + 0.5]
                if cands:
                    L["rect"][k] = min(cands, key=lambda o: abs(o - cur))
            for w in walls:                               # the floor covers at least the foot of its walls
                if w["dom"] != L["dom"] or abs(w["rect"][2] - L["off"]) > 0.05:
                    continue
                if w["axis"] == "F":
                    L["rect"][0] = min(L["rect"][0], w["rect"][0])
                    L["rect"][1] = max(L["rect"][1], w["rect"][1])
                    L["rect"][2] = min(L["rect"][2], w["off"])
                    L["rect"][3] = max(L["rect"][3], w["off"])
                else:
                    L["rect"][2] = min(L["rect"][2], w["rect"][0])
                    L["rect"][3] = max(L["rect"][3], w["rect"][1])
                    L["rect"][0] = min(L["rect"][0], w["off"])
                    L["rect"][1] = max(L["rect"][1], w["off"])

    def _quads(self, planes):
        quads = []
        for p in planes:
            A, B, Av = self.frame.of(p["axis"])
            R0 = p["rect"]
            if (R0[1] - R0[0]) * (R0[3] - R0[2]) < 1.0:
                continue
            rects = [tuple(R0)]
            for c in p.get("cuts", []):
                c = list(c)
                if c[2] - R0[2] < 0.8:                    # opening almost touching the bottom: door, down to the floor
                    c[2] = R0[2] - 0.01
                rects = [r for r in rects for r in rectSubtract(r, c)]

            def P3(b, a):
                d = p["off"] + p["yaw"] * (b - p.get("yaw_c", 0.0))
                return b * B + a * Av + d * A
            kind = p.get("name", "floor" if p["axis"] == "U" else "wall")
            name = "%s_%s_%s" % (p["dom"], kind, planeLabel(p["axis"], p["off"]))
            for j, (b0, b1, a0, a1) in enumerate(rects):
                if (b1 - b0) * (a1 - a0) < 0.05:
                    continue
                quads.append(dict(V=np.array([P3(b0, a0), P3(b1, a0), P3(b1, a1), P3(b0, a1)]),
                                  name=name + ("_%d" % j if len(rects) > 1 else ""), axis=p["axis"], off=p["off"],
                                  refined=p["src"] != "measured"))
            self.log("%-34s %6.1f u2 in %d pieces, %d openings (%s)" % (
                name, (R0[1] - R0[0]) * (R0[3] - R0[2]), len(rects), len(p.get("cuts", [])), p["src"]))
        return quads

    def run(self):
        """All the steps -> <work>/largePlanes.pkl (frame, planes, quads), largePlanes.obj, volumes.obj (+ stairs).

        Returns:
            dict: frame (F, S, U axes), planes, quads.
        """
        self._cloud()
        planes = self._fuse(self._measure())
        self.log("after fusion: %d large planes" % len(planes))
        self.log("photometric refinement")
        for pl in planes:
            self._refine(pl)
        # unverified small outdoor planes (trees, cornices seen from afar) dropped; fused again after refinement
        planes = [p for p in planes if not (p["dom"] == "exterior" and p["src"] == "measured" and p["support"] < 8)]
        planes = self._fuse(planes)
        self.log("after refinement: %d large planes" % len(planes))
        carved = {"exterior": [], "interior": []}
        for name, _, _, interior in self.domains:
            with open(os.path.join(self.work, name, "carve.pkl"), "rb") as f:
                carved["interior" if interior else "exterior"] += pickle.load(f)
        for pl in planes:
            pl["cuts"] = self._openings(pl, carved) if pl["axis"] != "U" and pl["support"] >= 5 else []
            if pl["cuts"]:
                self.log("openings %s %.2f (%s): %s" % (pl["axis"], pl["off"], pl["dom"],
                                                        [np.round(c, 2).tolist() for c in pl["cuts"]]))
        planes = self._removeGhosts(planes)
        if self.dense is not None:
            self._extendAlongDense(planes)
        stairs = loadStairs(self.work)
        self._levels(planes, stairs)
        self._floors(planes, stairs)
        self._joints(planes)
        quads = self._quads(planes)
        result = dict(quads=quads, up=self.frame.axes["U"], frame=dict(self.frame.axes),
                      planes=[{k: v for k, v in p.items() if k != "pts"} for p in planes])
        with open(os.path.join(self.work, LARGE_PLANES_FILE), "wb") as f:
            pickle.dump(result, f)
        writeObj(os.path.join(self.work, "largePlanes.obj"), [(q["name"], [q["V"]]) for q in quads])
        self.log("%d large plane quads" % len(quads))
        return result


def loadDensePoints(meshPath, scale, log=print):
    """Vertices of a Meshroom dense mesh (frame x, -y, -z of the SfM) in the working frame, or None."""
    if not meshPath or not os.path.isfile(meshPath):
        return None
    V = readObjVertices(meshPath) * MESHROOM_FLIP * scale
    log("dense mesh: %d vertices" % len(V))
    return V


def loadLargePlanes(work):
    with open(os.path.join(work, LARGE_PLANES_FILE), "rb") as f:
        return pickle.load(f)
