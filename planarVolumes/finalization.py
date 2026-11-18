"""Apply the photometric refinement to the carved planes, resolve overlaps (the most photo-consistent plane wins),
then maximal rectangles.

Indoors, failed refinements come from occlusions and are kept, and walls are extended down to the floor (their
bottom is hidden by people) except where seen-through votes exist (doors).
"""
import os
import pickle

import cv2
import numpy as np

from .geometry import maxRect, writeObj

FINAL_FILE = "final.pkl"


def _isFloor(g, up):
    return abs(g["p"]["n"] @ up) > 0.9


def _isMonocular(g):
    """Monocular-only candidate (no track point)."""
    return g["p"].get("src") == "moge" and g["p"]["ntrk"] == 0


def planePoint(g, u, v):
    p = g["p"]
    height = p["off"] + g["delta"] + g["tx"] * (u - g["uc"][0]) + g["ty"] * (v - g["uc"][1])
    return u[..., None] * p["b"] + v[..., None] * p["a"] + height[..., None] * p["n"]


def _cells(g):
    ys, xs = np.nonzero(g["mask"])
    u = g["lo"][0] + (xs + .5) * g["cell"]
    v = g["lo"][1] + (ys + .5) * g["cell"]
    return ys, xs, planePoint(g, u, v)


def _toGrid(g, X):
    p = g["p"]
    u, v = X @ p["b"], X @ p["a"]
    height = p["off"] + g["delta"] + g["tx"] * (u - g["uc"][0]) + g["ty"] * (v - g["uc"][1])
    xs = np.floor((u - g["lo"][0]) / g["cell"]).astype(int)
    ys = np.floor((v - g["lo"][1]) / g["cell"]).astype(int)
    return ys, xs, np.abs(X @ p["n"] - height)


def _cellScore(g, ys, xs):
    s = g["score"][ys, xs]
    return np.where(np.isfinite(s), s, -0.5 if not g["ok"] else 0.0)


def resolveOverlaps(planes, log=print):
    """Remove from each plane the overlap zones where another plane is clearly more photo-consistent."""
    remove = [np.zeros_like(g["mask"]) for g in planes]
    for ia, ga in enumerate(planes):
        ys, xs, X = _cells(ga)
        sa = _cellScore(ga, ys, xs)
        for ib, gb in enumerate(planes):
            if ia == ib or abs(ga["p"]["n"] @ gb["p"]["n"]) < np.cos(np.radians(30)):
                continue
            yb, xb, dist = _toGrid(gb, X)
            H, W = gb["mask"].shape
            inside = (yb >= 0) & (xb >= 0) & (yb < H) & (xb < W)
            tol = max(0.5, 0.06 * min(ga["zmed"], gb["zmed"]))
            conflict = np.zeros(len(ys), bool)
            conflict[inside] = gb["mask"][yb[inside], xb[inside]] & (dist[inside] < tol)
            if conflict.sum() < 30:
                continue
            # decision per overlap zone (mean scores), not per cell
            ma = np.nanmean(sa[conflict])
            mb = np.nanmean(_cellScore(gb, yb[conflict], xb[conflict]))
            # a monocular-only candidate takes a zone from an anchored plane only if clearly better
            margin = 0.02 + (0.06 if _isMonocular(gb) and not _isMonocular(ga) else 0) \
                - (0.06 if _isMonocular(ga) and not _isMonocular(gb) else 0)
            if mb > ma + margin or (abs(mb - ma) <= abs(margin) and gb["mask"].sum() > ga["mask"].sum() and margin == 0.02):
                remove[ia][ys[conflict], xs[conflict]] = True
    for g, r in zip(planes, remove):
        if _isMonocular(g) and (g["mask"] & ~r).sum() * g["cell"] ** 2 < 1.0:   # monocular candidate reduced to crumbs
            r |= g["mask"]
        n0 = g["mask"].sum()
        g["mask"] &= ~r
        m = cv2.morphologyEx(g["mask"].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        g["mask"] = m > 0
        if r.sum() > 200:
            log("plane %2d fam %2d: %5d -> %5d cells (overlaps lost)" % (g["i"], g["p"]["fam"], n0, g["mask"].sum()))


def extendWallsToFloor(planes, carved, up, log=print):
    """Indoors: walls go down to the main floor, except where seen-through votes exist (doors, windows)."""
    floors = [g for g in planes if abs(g["p"]["n"] @ up) > 0.9]
    if not floors:
        return
    fl = max(floors, key=lambda g: g["mask"].sum())
    hf = (fl["p"]["off"] + fl["delta"]) * (fl["p"]["n"] @ up)
    log("floor at height %.2f (plane %d)" % (hf, fl["i"]))
    for g in planes:
        if abs(g["p"]["n"] @ up) > 0.2 or g["mask"].sum() < 300:
            continue
        o = carved[g["i"]]
        cell = g["cell"]
        if abs(g["p"]["b"] @ up) > abs(g["p"]["a"] @ up):
            continue                                     # vertical axis expected along the rows (a)
        H, W = g["mask"].shape
        Vv = g["lo"][1] + (np.arange(-200, H + 200) + .5) * cell
        Uu = g["lo"][0] + (np.arange(W) + .5) * cell
        heights = planePoint(g, Uu[None, :].repeat(len(Vv), 0), Vv[:, None].repeat(W, 1)) @ up
        new = np.zeros((H + 400, W), bool)
        new[200:200 + H] = g["mask"]
        Fp = np.zeros_like(new, np.float32)
        Fp[200:200 + H] = o["F"]
        Sp = np.zeros_like(new, np.float32)
        Sp[200:200 + H] = o["S"]
        added = 0
        for x in range(W):
            col = np.nonzero(new[:, x])[0]
            if len(col) < 3:
                continue
            low = col[np.argmin(heights[col, x])]
            step = -1 if heights[low - 1, x] < heights[low, x] else 1 if low + 1 < len(Vv) else 0
            r = low + step
            while 0 <= r < len(Vv) and heights[r, x] > hf and not (Fp[r, x] >= 2 and Fp[r, x] > Sp[r, x]):
                new[r, x] = True
                added += 1
                r += step
        if added:
            g["mask"] = new
            g["lo"] = np.array([g["lo"][0], g["lo"][1] - 200 * cell])
            g["score"] = np.full(new.shape, np.nan)
            log("wall %d: +%d cells down to the floor" % (g["i"], added))


def finalizeDomain(domainDir, interior, log=print):
    """Refined planes of a domain -> <domainDir>/final.pkl, final.obj.

    Args:
        interior: indoor domain (keep the planes whose refinement failed, extend the walls to the floor).
    """
    with open(os.path.join(domainDir, "carve.pkl"), "rb") as f:
        carved = pickle.load(f)
    with open(os.path.join(domainDir, "refined.pkl"), "rb") as f:
        refined = pickle.load(f)
    with open(os.path.join(domainDir, "carved.pkl"), "rb") as f:
        up = pickle.load(f)["up"]
    planes = []
    for i, (o, rf) in enumerate(zip(carved, refined)):
        if o is None or o["mask"].sum() < 100:
            continue
        p = o["plane"]
        g = dict(i=i, p=p, mask=o["mask"].copy() > 0, lo=o["lo"], cell=o["cell"], zmed=o["zmed"], delta=0.0, tx=0.0, ty=0.0,
                 uc=(0, 0), score=np.full(o["mask"].shape, np.nan), ok=False, tried=rf is not None)
        thresholds = (0.4, 0.1) if p.get("src") == "moge" and p["ntrk"] == 0 else (0.25, 0.04)
        if rf is not None and rf["peak"] > thresholds[0] and rf["contrast"] > thresholds[1]:
            tiltOk = rf["peak"] - rf["flat"] > 0.03 and max(abs(np.degrees(np.arctan(rf["tx"]))),
                                                            abs(np.degrees(np.arctan(rf["ty"])))) < 4.9
            g.update(delta=rf["delta"], uc=rf["uc"], ok=True, score=rf["cellscore"])
            if tiltOk:
                g.update(tx=rf["tx"], ty=rf["ty"])
        planes.append(g)
    # poorly photo-consistent planes (vegetation, false planes) dropped, except floors and ceilings (grazing views);
    # monocular-only candidates are kept only if validated by the refinement
    drop = [g["i"] for g in planes if (not _isFloor(g, up) and g["tried"] and not g["ok"] and not interior)
            or (_isMonocular(g) and not g["ok"])]
    planes = [g for g in planes if g["i"] not in drop]
    log("%s: %d planes kept, dropped (poorly photo-consistent): %s" % (os.path.basename(domainDir), len(planes), drop))
    resolveOverlaps(planes, log)
    if interior:
        extendWallsToFloor(planes, carved, up, log)
    quads = []
    for g in planes:
        mm = g["mask"].astype(np.uint8).copy()
        cell, lo = g["cell"], g["lo"]
        for _ in range(25):
            area, x0, y0, x1, y1 = maxRect(mm)
            if area * cell ** 2 < 0.5:
                break
            mm[y0:y1, x0:x1] = 0
            u0, v0 = lo + np.array([x0, y0]) * cell
            u1, v1 = lo + np.array([x1, y1]) * cell
            V = planePoint(g, np.array([u0, u1, u1, u0]), np.array([v0, v0, v1, v1]))
            quads.append(dict(V=V, fam=g["p"]["fam"], plane=g["i"], off=g["p"]["off"] + g["delta"], area=area * cell ** 2,
                              frames=g["p"]["frames"], ntrk=g["p"]["ntrk"], refined=g["ok"]))
    with open(os.path.join(domainDir, FINAL_FILE), "wb") as f:
        pickle.dump(dict(quads=quads, up=up, planes=planes), f)
    writeObj(os.path.join(domainDir, "final.obj"),
             [("plane%03d_q%03d_fam%02d%s" % (q["plane"], k, q["fam"], "" if q["refined"] else "_unrefined"), [q["V"]])
              for k, q in enumerate(quads)])
    log("  %d rectangles" % len(quads))
    return planes
