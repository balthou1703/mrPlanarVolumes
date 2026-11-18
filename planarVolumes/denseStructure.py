"""Structure from the dense mesh, in the dominant frame of the large planes: missing large planes (turrets,
retaining walls, terrace, balustrade) and stairs as individual steps.

1. planes: for each axis A of the frame, area-weighted histogram of the position c.A of the surface points oriented
   along A; each peak -> band -> occupancy grid in the plane -> components -> rectangles (greedy decomposition).
   The planes duplicating the large planes are dropped.
2. stairs (any shape, straight or turning): the treads are horizontal surfaces at regular heights. Zone around the
   straight stairs of the monocular step, comb on the heights of the horizontal points, footprint of each tread
   chained from the landing, one prism per step from the foot to its tread.
"""
import os
import pickle

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .geometry import comb, greedyRects, writeObj
from .largePlanes import INPLANE

DENSE_STRUCTURE_FILE = "denseStructure.pkl"
COS = np.cos(np.radians(20))
BIN = 0.03          # histogram bin (u)
BAND = 0.08         # half thickness of a plane (u)
CELL = 0.10         # occupancy grid (u)
MIN_PEAK = 0.6      # smallest area of a plane peak (u2)
MIN_RECT = 0.6      # smallest plane rectangle (u2)
DUPLICATE_OFFSET = 0.3   # position gap under which a plane duplicates an existing one (u)
VOXEL = 0.03        # surface resampling (u)
RADIUS = 0.12       # neighbourhood of the normal estimation (u)
STEP_CELL = 0.02    # grid of the tread footprints (u)
TREAD_RADIUS = 0.06  # normals of the shallow treads: smaller than RADIUS, which overlaps the step noses


def histogramPeaks(x, w, minWeight):
    lo = x.min()
    n = int((x.max() - lo) / BIN) + 2
    h = np.bincount(((x - lo) / BIN).astype(int), w, n)
    hs = ndimage.gaussian_filter1d(h, 1.0)
    out = []
    for i in np.argsort(-hs):
        if hs[i] * 2.5 < minWeight:                     # ~ area within +-BAND
            break
        if all(abs(i - j) * BIN > 2 * BAND for j in out):
            out.append(i)
    return [lo + (i + 0.5) * BIN for i in out]


def resampleSurface(V, F, rng):
    """Regular surface points (voxel centroids of random samples of the triangles), with local PCA normals.

    Returns:
        tuple: points (n, 3), normals (n, 3), planar (n,) boolean (locally planar neighbourhood).
    """
    T = V[F]
    A2 = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    perFace = np.minimum(np.ceil(0.5 * A2 / (VOXEL * VOXEL) * 2).astype(int), 50)
    fi = np.repeat(np.arange(len(F)), perFace)
    r1, r2 = rng.random(len(fi)), rng.random(len(fi))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    S = T[fi, 0] + r1[:, None] * (T[fi, 1] - T[fi, 0]) + r2[:, None] * (T[fi, 2] - T[fi, 0])
    key = np.floor(S / VOXEL).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    Cc = np.zeros((len(cnt), 3))
    np.add.at(Cc, inv, S)
    Cc /= cnt[:, None]
    neighbours = cKDTree(Cc).query_ball_point(Cc, RADIUS, workers=-1)
    N = np.zeros_like(Cc)
    ok = np.zeros(len(Cc), bool)
    for i, js in enumerate(neighbours):
        if len(js) < 8:
            continue
        Q = Cc[js] - Cc[js].mean(0)
        w, vec = np.linalg.eigh(Q.T @ Q)
        if w[0] < 0.15 * w[1]:
            N[i] = vec[:, 0]
            ok[i] = True
    return Cc, N, ok


def missingPlanes(Cc, N, ok, axes, existing, log=print):
    """Rectangles of the dense planes that do not duplicate an existing large plane (axis, offset)."""
    area = np.full(len(Cc), VOXEL * VOXEL)
    quads = []
    for axis in ("F", "S", "U"):
        A = axes[axis]
        B, Av = axes[INPLANE[axis][0]], axes[INPLANE[axis][1]]
        sel = ok & (np.abs(N @ A) > COS)
        if not sel.any():
            continue
        off = Cc[sel] @ A
        peaks = histogramPeaks(off, area[sel], MIN_PEAK if axis != "U" else 0.15)
        log("axis %s: %d oriented points, %d peaks" % (axis, sel.sum(), len(peaks)))
        idx = np.flatnonzero(sel)
        for d in sorted(peaks):
            band = idx[np.abs(off - d) < BAND]
            if len(band) < 30:
                continue
            u, v = Cc[band] @ B, Cc[band] @ Av
            u0, v0 = u.min(), v.min()
            gu, gv = ((u - u0) / CELL).astype(int), ((v - v0) / CELL).astype(int)
            occ = np.zeros((gv.max() + 1, gu.max() + 1))
            np.add.at(occ, (gv, gu), area[band])
            lab, nl = ndimage.label(ndimage.binary_closing(occ > 0.25 * CELL * CELL, iterations=1))
            duplicate = any(ax == axis and abs(o - d) < DUPLICATE_OFFSET for ax, o in existing)
            for li in range(1, nl + 1):
                cm = lab == li
                if cm.sum() * CELL * CELL < MIN_RECT or duplicate:
                    continue
                for r0, r1, c0, c1 in greedyRects(cm):
                    ra = (u0 + c0 * CELL, u0 + c1 * CELL)
                    rb = (v0 + r0 * CELL, v0 + r1 * CELL)
                    if (ra[1] - ra[0]) * (rb[1] - rb[0]) < MIN_RECT:
                        continue
                    P0 = d * A
                    Q = np.array([P0 + ra[0] * B + rb[0] * Av, P0 + ra[1] * B + rb[0] * Av,
                                  P0 + ra[1] * B + rb[1] * Av, P0 + ra[0] * B + rb[1] * Av])
                    kind = "floor" if axis == "U" else "wall"
                    quads.append(dict(V=Q, name="dense_%s_%s%s%.1f" % (kind, axis, "m" if d < 0 else "p", abs(d))))
    log("planes added from the dense mesh: %d rectangles" % len(quads))
    return quads


def stairsFromLevels(Cc, ok, axes, stairs, log=print):
    """Steps of any stairs (straight or turning) around the straight stairs of the monocular step.

    1. zone: around the straight stairs, between its foot (ground) and its landing (terrace);
    2. rise and phase: comb on the heights of the horizontal points;
    3. per level: footprint (grid, components), chained from the landing down, then upwards by prediction;
    4. one step = prism of the footprint (simplified convex hull) from the foot to the tread.

    Returns:
        list of quads (one prism per step).
    """
    if stairs is None:
        log("stairs (dense): no straight stairs, skipped")
        return []
    U, Fv, Sv = axes["U"], axes["F"], axes["S"]
    SV = np.array([q["V"] for q in stairs["quads"]]).reshape(-1, 3)
    sb = np.c_[SV @ Fv, SV @ Sv]
    hFoot, hTop = (SV @ U).min(), (SV @ U).max()
    pf, ps, ph = Cc @ Fv, Cc @ Sv, Cc @ U
    R = 2.5
    zone = (pf > sb[:, 0].min() - R) & (pf < sb[:, 0].max() + R) & (ps > sb[:, 1].min() - R) & (ps < sb[:, 1].max() + R)
    iz = np.flatnonzero(zone)
    Z = Cc[iz]
    horizontal = np.zeros(len(Cc), bool)
    for j, js in enumerate(cKDTree(Z).query_ball_point(Z, TREAD_RADIUS, workers=-1)):
        if len(js) >= 5:
            Q = Z[js] - Z[js].mean(0)
            w, vec = np.linalg.eigh(Q.T @ Q)
            horizontal[iz[j]] = w[0] < 0.2 * w[1] and abs(vec[:, 0] @ U) > COS
    hz = horizontal & (ph > hFoot + 0.03) & (ph < hTop + 1.0)
    if hz.sum() < 200:
        log("stairs (dense): too few horizontal surfaces (%d)" % hz.sum())
        return []
    rise0 = float(stairs["params"].get("step_h", 0.1))
    hc = hz & (ph < hTop - 0.03)
    concentration, rise, phase = comb(ph[hc], np.ones(hc.sum()), 0.6 * rise0, 1.6 * rise0)
    levels = [lv for lv in phase + rise * np.arange(int((hFoot - phase) / rise) - 2, int((hTop - phase) / rise) + 3)
              if hFoot + 0.5 * rise < lv < hTop - 0.4 * rise]
    log("stairs (dense): rise %.3f u (R %.2f), %d levels between %.2f and %.2f" % (
        rise, concentration, len(levels), hFoot, hTop))
    if concentration < 0.15 or len(levels) < 2:
        log("stairs (dense): no regular heights")
        return []
    f0, s0 = pf[zone].min(), ps[zone].min()
    Wg = int((pf[zone].max() - f0) / STEP_CELL) + 2
    Hg = int((ps[zone].max() - s0) / STEP_CELL) + 2

    def grid(sel):
        g = np.zeros((Hg, Wg), np.uint8)
        g[((ps[sel] - s0) / STEP_CELL).astype(int), ((pf[sel] - f0) / STEP_CELL).astype(int)] = 1
        return cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    seed = np.zeros((Hg, Wg), np.uint8)              # footprint of the straight stairs
    cv2.fillConvexPoly(seed, cv2.convexHull(np.round(np.c_[(sb[:, 0] - f0), (sb[:, 1] - s0)] / STEP_CELL).astype(np.int32)), 1)
    nearPx = int(0.3 / STEP_CELL)
    K = np.ones((2 * nearPx + 1,) * 2, np.uint8)

    def tread(level, prev):
        """Tread at a level connected to the (dilated) footprint prev -> (mask, area, outline) or None."""
        g = grid(hz & (np.abs(ph - level) < 0.3 * rise))
        nl, lab, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
        best = None
        for i in range(1, nl):
            if stats[i, cv2.CC_STAT_AREA] * STEP_CELL ** 2 < 0.03:
                continue
            ov = ((lab == i) & (prev > 0)).sum()
            if ov and (best is None or ov > best[0]):
                best = (ov, i)
        if best is None:
            return None
        m = (lab == best[1]).astype(np.uint8)
        # the tread is only horizontal in pieces: extend it to the whole surface at this height (all the points of the
        # slice) within 0.35 u of the horizontal core
        grow = int(0.35 / STEP_CELL)
        band = grid(zone & (np.abs(ph - level) < 0.35 * rise))
        m = (band & cv2.dilate(m, np.ones((2 * grow + 1,) * 2, np.uint8))) | m
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        nl2, lab2, st2, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if nl2 < 2:
            return None
        m = (lab2 == 1 + np.argmax(st2[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        hull = cv2.approxPolyDP(cv2.convexHull(cv2.findNonZero(m)), 0.03 / STEP_CELL, True)[:, 0]
        return m, m.sum() * STEP_CELL ** 2, hull

    # 1. from the presumed landing down to the foot
    treads = []
    prev = cv2.dilate(seed, K)
    for level in sorted(levels, reverse=True):
        r = tread(level, prev)
        if r is None:
            log("  level %.3f: no connected tread" % level)
            continue
        treads.append((level, r[2], r[1]))
        log("  level %.3f: tread %.2f u2, %d vertices" % (level, r[1], len(r[2])))
        prev = cv2.dilate(r[0], K)

    # 2. above: the landing of the straight stairs may be too low. The climb is continued by predicting the next
    #    tread (displacement between the two top treads); a tread is kept only if its center falls near the prediction.
    #    A missing level between two found treads is filled with the previous tread translated.
    def center(h):
        return h.reshape(-1, 2).mean(0)
    if len(treads) >= 2:
        top = sorted(treads, key=lambda t: -t[0])
        d = center(top[0][1]) - center(top[1][1])
        level, hull, area = top[0]
        pending, extra = [], []
        while len(pending) < 2 and level < hTop + 1.0:
            level += rise
            predicted = hull + np.round(d).astype(int)
            pm = np.zeros((Hg, Wg), np.uint8)
            cv2.fillConvexPoly(pm, predicted.astype(np.int32), 1)
            r = tread(level, cv2.dilate(pm, K))
            found = r is not None and r[1] < 3 * area and np.linalg.norm(center(r[2]) - center(predicted)) * STEP_CELL < 0.3
            if not found:
                pending.append((level, predicted, area))
                hull = predicted
                log("  level %.3f: no tread at the predicted position" % level)
                continue
            extra += pending
            pending = []
            d = 0.5 * d + 0.5 * (center(r[2]) - center(hull - np.round(d).astype(int)))
            hull, area = r[2], r[1]
            extra.append((level, hull, area))
            log("  level %.3f: tread %.2f u2 (above the presumed landing)" % (level, area))
        treads += extra
    if len(treads) < 3:
        log("stairs (dense): fewer than 3 connected treads")
        return []
    med = np.median([t[2] for t in treads])               # too large = floor or planter swallowed
    treads = [t for t in treads if t[2] < 4 * med]
    out = []
    for k, (level, hull, _) in enumerate(sorted(treads)):
        P2 = np.c_[f0 + (hull[:, 0] + 0.5) * STEP_CELL, s0 + (hull[:, 1] + 0.5) * STEP_CELL]
        top = [p[0] * Fv + p[1] * Sv + level * U for p in P2]
        bottom = [p[0] * Fv + p[1] * Sv + hFoot * U for p in P2]
        if np.cross(top[1] - top[0], top[2] - top[0]) @ U < 0:     # top face normal towards +U
            top, bottom = top[::-1], bottom[::-1]
        n = len(top)
        faces = [top, bottom[::-1]] + [[bottom[q], bottom[(q + 1) % n], top[(q + 1) % n], top[q]] for q in range(n)]
        out += [dict(V=np.array(f), name="dense_stairs_step%02d" % k) for f in faces]
    log("stairs (dense): %d steps, rise %.3f u, foot %.2f, landing %.2f" % (len(treads), rise, hFoot, hTop))
    return out


def computeDenseStructure(work, large, stairs, meshPath, scale, log=print):
    """Dense structure -> <work>/denseStructure.pkl (quads, stairs found), denseStructure.obj.

    Returns:
        dict: quads, stairs (bool: the dense stairs replace the straight ones); None without dense mesh.
    """
    from .geometry import MESHROOM_FLIP, readObj
    if not meshPath or not os.path.isfile(meshPath):
        log("no dense mesh: dense structure skipped")
        return None
    axes = {k: np.asarray(v, float) for k, v in large["frame"].items()}
    V, F = readObj(meshPath)
    V = V * MESHROOM_FLIP * scale
    log("dense mesh: %d vertices, %d faces" % (len(V), len(F)))
    Cc, N, ok = resampleSurface(V, F, np.random.default_rng(0))
    log("resampled surface: %d points, %d locally planar" % (len(Cc), ok.sum()))
    quads = missingPlanes(Cc, N, ok, axes, [(p["axis"], p["off"]) for p in large["planes"]], log)
    stairsQuads = stairsFromLevels(Cc, ok, axes, stairs, log)
    result = dict(quads=quads + stairsQuads, up=axes["U"], stairs=bool(stairsQuads))
    with open(os.path.join(work, DENSE_STRUCTURE_FILE), "wb") as f:
        pickle.dump(result, f)
    writeObj(os.path.join(work, "denseStructure.obj"), [(q["name"], [q["V"]]) for q in result["quads"]])
    return result
