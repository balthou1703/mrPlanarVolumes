"""Openings of a large wall as simple shapes (CSG): the wall stays ONE slab, the measured recesses are recognized as
primitives (rectangle, arch: semicircular or segmental, circle) chosen by model selection (true minus false cells
penalized by the number of parameters), regularized (common alignments and widths), subtracted from the slab and
extruded inwards (reveals + back = glazing; through opening without back). Few vertices: quads and fans.
"""
import os

import cv2
import numpy as np

from .geometry import planeLabel, rectsOverlap
from .recesses import planeAxes, planeById

# recess depth (u): a cell is recessed beyond THRESHOLD, or beyond THRESHOLD_LOW when it touches a zone beyond
# THRESHOLD (hysteresis: a slightly recessed window stays in one piece, a transom joins its door)
THRESHOLD = 0.15
THRESHOLD_LOW = 0.08
MIN_WIDTH, MIN_HEIGHT, MIN_AREA = 0.4, 0.5, 0.2     # smallest opening kept (u)
THROUGH_DEPTH = 1.3     # median depth beyond which the opening goes through the wall
WALL_THICKNESS = 0.6    # depth of the reveals of a through opening (u)
ARC_SEGMENTS = 12
NB_PARAMS = {"rect": 4, "arch": 5, "circle": 3}


def archTop(x, x0, x1, ys, rho):
    """Top of an arch (chord x0-x1, rise rho, springing height ys); rho = 0 -> straight lintel."""
    w = x1 - x0
    xc = (x0 + x1) / 2
    if rho < 1e-3:
        return np.full_like(x, ys)
    R = (rho ** 2 + (w / 2) ** 2) / (2 * rho)
    return ys + rho - R + np.sqrt(np.maximum(R ** 2 - (x - xc) ** 2, 0))


def rasterize(pr, bb, aa):
    """Boolean grid (rows aa, columns bb) of the cells inside a primitive."""
    B, A = np.meshgrid(bb, aa)
    if pr["type"] == "circle":
        return (B - pr["cx"]) ** 2 + (A - pr["cy"]) ** 2 <= pr["r"] ** 2
    x0, x1, y0, ys, rho = pr["x0"], pr["x1"], pr["y0"], pr["ys"], pr["rho"]
    return (B >= x0) & (B <= x1) & (A >= y0) & (A <= archTop(B, x0, x1, ys, rho))


def candidates(m, b, a, cell):
    """Primitive hypotheses from the profiles of a component (data-driven, RANSAC like)."""
    out = []
    colFill = m.mean(0)
    for thr in (0.15, 0.3, 0.5):
        on = colFill > thr * colFill.max()
        j = 0
        while j < len(on):                               # contiguous runs of recessed columns
            if not on[j]:
                j += 1
                continue
            k = j
            while k + 1 < len(on) and on[k + 1]:
                k += 1
            if (k - j + 1) * cell >= MIN_WIDTH:
                x0, x1 = b[j] - cell / 2, b[k] + cell / 2
                sub = m[:, j:k + 1]
                full = np.nonzero(sub.mean(1) > 0.5)[0]
                if len(full):
                    y0 = a[full[0]] - cell / 2
                    top = np.array([a[np.nonzero(sub[:, q])[0].max()] + cell / 2 if sub[:, q].any() else np.nan
                                    for q in range(sub.shape[1])])
                    bx = b[j:k + 1]
                    ok = np.isfinite(top)
                    for rho in np.arange(0, (x1 - x0) / 2 + 1e-6, 0.05):
                        ys = float(np.median(top[ok] - archTop(bx[ok], x0, x1, 0.0, rho)))
                        if ys - y0 < 0.5 * MIN_HEIGHT:
                            continue
                        out.append(dict(type="arch" if rho > 0.1 else "rect", x0=x0, x1=x1, y0=y0, ys=ys,
                                        rho=rho if rho > 0.1 else 0.0))
            j = k + 1
    ys_, xs_ = np.nonzero(m)
    out.append(dict(type="circle", cx=float(b[xs_].mean()), cy=float(a[ys_].mean()),
                    r=float(np.sqrt(len(xs_) * cell ** 2 / np.pi))))
    return out


def fitSequential(mask, bb, aa, cell, maxPrimitives=6):
    """Sequential extraction: best primitive (true recessed cells - 2 x false ones, cost of the parameters), remove
    the explained cells, start again on the rest (Efficient RANSAC, Schnabel et al. 2007)."""
    ys_, xs_ = np.nonzero(mask)
    sub = (slice(max(ys_.min() - 4, 0), ys_.max() + 5), slice(max(xs_.min() - 4, 0), xs_.max() + 5))
    m = mask[sub].copy()
    b, a = bb[sub[1]], aa[sub[0]]
    res = []
    for _ in range(maxPrimitives):
        if m.sum() * cell ** 2 < MIN_AREA:
            break
        bestPerType = {}
        for pr in candidates(m, b, a, cell):
            R = rasterize(pr, b, a)
            sc = (R & m).sum() - 2 * (R & ~m).sum()
            if sc > bestPerType.get(pr["type"], (None, 0))[1]:
                bestPerType[pr["type"]] = (pr, sc)
        if not bestPerType:
            break
        # model selection: a richer shape must win clearly (3 % per additional parameter)
        best, bs = None, 0
        for t in ("circle", "rect", "arch"):
            extra = NB_PARAMS[t] - NB_PARAMS[best["type"]] if best else 0
            if t in bestPerType and bestPerType[t][1] > bs * (1 + 0.03 * extra):
                best, bs = bestPerType[t]
        if best is None:
            break
        R = rasterize(best, b, a)
        tp = (R & m).sum()
        precision = tp / max(R.sum(), 1)
        if precision < 0.7 or tp * cell ** 2 < MIN_AREA:
            break
        best["iou"] = float(tp / max((R | m).sum(), 1))
        best["precision"] = float(precision)
        res.append(best)
        m &= ~cv2.dilate(R.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return res


def regularize(P, tol=0.4, tolWidth=0.35):
    """Facade alignments: close sills, springings, tops and widths -> common value."""
    for key in ("y0", "ys", "top"):
        vals = sorted([(p["ys"] + p["rho"] if key == "top" else p[key], p) for p in P if p["type"] != "circle"],
                      key=lambda t: t[0])
        groups, cur = [], []
        for v, p in vals:
            if cur and v - cur[-1][0] > tol:
                groups.append(cur)
                cur = []
            cur.append((v, p))
        if cur:
            groups.append(cur)
        for g in groups:
            if len(g) < 2:
                continue
            m = float(np.median([v for v, _ in g]))
            for v, p in g:
                if key == "top":
                    p["ys"] += m - v
                else:
                    p[key] = m
    # widths: same family within tolWidth -> median width, center kept
    rec = sorted([p for p in P if p["type"] != "circle"], key=lambda p: p["x1"] - p["x0"])
    i = 0
    while i < len(rec):
        j = i
        while j + 1 < len(rec) and (rec[j + 1]["x1"] - rec[j + 1]["x0"]) - (rec[i]["x1"] - rec[i]["x0"]) < tolWidth:
            j += 1
        if j > i:
            wm = float(np.median([p["x1"] - p["x0"] for p in rec[i:j + 1]]))
            for p in rec[i:j + 1]:
                xc = (p["x0"] + p["x1"]) / 2
                p["x0"], p["x1"] = xc - wm / 2, xc + wm / 2
                p["rho"] = min(p["rho"], wm / 2)
        i = j + 1


def outline(p):
    """Counter-clockwise outline (b, a) of a primitive."""
    if p["type"] == "circle":
        t = np.linspace(0, 2 * np.pi, 2 * ARC_SEGMENTS, endpoint=False)
        return np.stack([p["cx"] + p["r"] * np.cos(t), p["cy"] + p["r"] * np.sin(t)], 1)
    x0, x1, y0, ys, rho = p["x0"], p["x1"], p["y0"], p["ys"], p["rho"]
    pts = [(x0, y0), (x1, y0)]
    if rho > 0:
        pts += [(x, float(archTop(np.array([x]), x0, x1, ys, rho)[0])) for x in np.linspace(x1, x0, ARC_SEGMENTS + 1)]
    else:
        pts += [(x1, ys), (x0, ys)]
    return np.array(pts)


def bbox(p):
    if p["type"] == "circle":
        return p["cx"] - p["r"], p["cx"] + p["r"], p["cy"] - p["r"], p["cy"] + p["r"]
    return p["x0"], p["x1"], p["y0"], p["ys"] + p["rho"]


def holedSlab(rect, P):
    """Slab rect minus the primitives: grid of the bounding boxes (maximal rectangles) + spandrels as fans.

    Returns:
        list of (b, a) polygons, each convex or star-shaped from its first vertex.
    """
    b0, b1, a0, a1 = rect
    xs = sorted({b0, b1} | {min(max(v, b0), b1) for p in P for v in bbox(p)[:2]})
    ys = sorted({a0, a1} | {min(max(v, a0), a1) for p in P for v in bbox(p)[2:]})
    nx, ny = len(xs) - 1, len(ys) - 1
    free = np.ones((ny, nx), bool)
    for p in P:
        x0, x1, y0, y1 = bbox(p)
        for j in range(nx):
            for i in range(ny):
                cx, cy = (xs[j] + xs[j + 1]) / 2, (ys[i] + ys[i + 1]) / 2
                if x0 < cx < x1 and y0 < cy < y1:
                    free[i, j] = False
    polys = []
    used = np.zeros_like(free)
    for i in range(ny):                                  # greedy merge: horizontal runs, then vertical extension
        j = 0
        while j < nx:
            if not free[i, j] or used[i, j]:
                j += 1
                continue
            k = j
            while k + 1 < nx and free[i, k + 1] and not used[i, k + 1]:
                k += 1
            last = i
            while last + 1 < ny and free[last + 1, j:k + 1].all() and not used[last + 1, j:k + 1].any():
                last += 1
            used[i:last + 1, j:k + 1] = True
            polys.append(np.array([(xs[j], ys[i]), (xs[k + 1], ys[i]), (xs[k + 1], ys[last + 1]), (xs[j], ys[last + 1])]))
            j = k + 1
    for p in P:                                          # bounding box minus the shape
        x0, x1, y0, y1 = bbox(p)
        if p["type"] == "arch":
            xc = (p["x0"] + p["x1"]) / 2
            left = [(x, float(archTop(np.array([x]), p["x0"], p["x1"], p["ys"], p["rho"])[0]))
                    for x in np.linspace(x0, xc, ARC_SEGMENTS // 2 + 1)]
            right = [(x, float(archTop(np.array([x]), p["x0"], p["x1"], p["ys"], p["rho"])[0]))
                     for x in np.linspace(xc, x1, ARC_SEGMENTS // 2 + 1)]
            polys.append(np.array([(x0, y1)] + [(x0, p["ys"])] + left[1:]))      # left spandrel (fan from the corner)
            polys.append(np.array([(x1, y1)] + right[:-1] + [(x1, p["ys"])]))
        elif p["type"] == "circle":
            t = np.linspace(0, 2 * np.pi, 2 * ARC_SEGMENTS + 1)
            ring = np.stack([p["cx"] + p["r"] * np.cos(t), p["cy"] + p["r"] * np.sin(t)], 1)
            q = len(t) // 4
            for k, corner in enumerate([(x1, y1), (x0, y1), (x0, y0), (x1, y0)]):
                seg = ring[k * q:(k + 1) * q + 1]
                polys.append(np.array([corner] + [tuple(s) for s in seg[::-1]]))
    return polys


def crossings(scene, p, A, B, Av, maxJump=1.0):
    """Points (b, a) where the camera path (consecutive frames) crosses the plane p."""
    C = scene.centers()
    d = C @ A - p["off"]
    out = []
    for i in np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0):
        t = d[i] / (d[i] - d[i + 1])
        X = C[i] + t * (C[i + 1] - C[i])
        if np.linalg.norm(C[i + 1] - C[i]) < maxJump:        # pose jump: not a crossing
            out.append((float(X @ B), float(X @ Av)))
    return out


def wallOpenings(scene, pid, large, M, cameraSide, log=print):
    """Primitives of a wall and its CSG meshes.

    Args:
        pid: plane identifier.
        M: recess map of the wall.
        cameraSide: +1 / -1, side of the cameras relative to the plane.

    Returns:
        tuple: (objects [(name, polygons)], primitives, rect).
    """
    p = planeById(large, pid)
    A, B, Av = planeAxes(large, p)
    med = M["med"]
    cell = float(M["cell"])
    b0, b1, a0, a1 = [float(v) for v in M["rect"]]
    nb, na = med.shape[1], med.shape[0]
    bb = b0 + (np.arange(nb) + .5) * cell
    aa = a0 + (np.arange(na) + .5) * cell
    d0 = np.nan_to_num(med, nan=0)
    strong = cv2.morphologyEx((d0 > THRESHOLD).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    weak = cv2.morphologyEx((d0 > THRESHOLD_LOW).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, labWeak = cv2.connectedComponents(weak)
    keep = np.unique(labWeak[(strong > 0) & (labWeak > 0)])
    holes = np.isin(labWeak, keep).astype(np.uint8)
    holes = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))        # joins the leaves of a window
    nlab, lab, st, _ = cv2.connectedComponentsWithStats(holes)
    P = []
    for L in range(1, nlab):
        x, y, w, h, area = st[L]
        if area * cell ** 2 < MIN_AREA or w * cell < MIN_WIDTH or h * cell < MIN_HEIGHT:
            continue
        m = lab == L
        for pr in fitSequential(m, bb, aa, cell):
            if pr["type"] != "circle" and (pr["x1"] - pr["x0"] < MIN_WIDTH or pr["ys"] + pr["rho"] - pr["y0"] < MIN_HEIGHT):
                continue
            x0, x1, y0, y1 = bbox(pr)                    # touching a side of the slab: outside the wall, not a recess
            if x0 <= b0 + 2 * cell or x1 >= b1 - 2 * cell or (y1 >= a1 - 2 * cell and y0 <= a0 + 2 * cell):
                continue
            pr["depth"] = float(np.nanmedian(med[rasterize(pr, bb, aa) & m]))
            pr["through"] = pr["depth"] > THROUGH_DEPTH
            P.append(pr)
    # hidden bottom (vegetation, furniture in front): if the wall is not seen ON the plane below the opening, the
    # opening goes down to the floor (French window)
    for pr in P:
        if pr["type"] == "circle" or pr["y0"] - a0 < 0.05 or pr["y0"] - a0 > 3.0:      # lower level openings only
            continue
        cb = (bb > pr["x0"]) & (bb < pr["x1"])
        ca = (aa < pr["y0"]) & (aa > pr["y0"] - 1.2)
        zone = med[np.ix_(ca, cb)]
        onPlane = np.isfinite(zone) & (np.abs(zone) < THRESHOLD)
        other = any(q is not pr and q["type"] != "circle" and q["x1"] > pr["x0"] and q["x0"] < pr["x1"] and q["y0"] < pr["y0"]
                    for q in P)                          # another opening below: do not go down
        if pr["y0"] - a0 < 0.6 or (onPlane.mean() < 0.4 and not other):
            pr["y0"] = a0
    # opening crossed by the camera (the path goes through the plane inside it): necessarily a through opening
    for cb, ca in crossings(scene, p, A, B, Av):
        for pr in P:
            x0, x1, y0, y1 = bbox(pr)
            if x0 - 0.2 < cb < x1 + 0.2 and y0 - 0.2 < ca < y1 + 0.2 and not pr["through"]:
                pr["through"] = True
                pr["crossed"] = True
                log("  opening crossed by the camera at b %.2f a %.2f" % (cb, ca))
    regularize(P)
    for pr in P:                                         # stay inside the slab
        if pr["type"] != "circle":
            pr["x0"], pr["x1"], pr["y0"] = max(pr["x0"], b0), min(pr["x1"], b1), max(pr["y0"], a0)
    # known cuts of the large plane (stairs notch, seen-through openings): through rectangles without reveal, unless
    # a measured primitive already takes the place
    cuts = [list(c) for c in p.get("cuts", [])]
    merged = True
    while merged:                                        # overlapping cuts -> one (union)
        merged = False
        for i in range(len(cuts)):
            for j in range(i + 1, len(cuts)):
                if rectsOverlap(cuts[i], cuts[j]):
                    ci, cj = cuts[i], cuts.pop(j)
                    cuts[i] = [min(ci[0], cj[0]), max(ci[1], cj[1]), min(ci[2], cj[2]), max(ci[3], cj[3])]
                    merged = True
                    break
            if merged:
                break
    for c in cuts:
        c = [max(c[0], b0), min(c[1], b1), max(c[2], a0), min(c[3], a1)]
        if c[1] - c[0] < 0.3 or c[3] - c[2] < 0.3 or any(rectsOverlap(c, bbox(q)) for q in P):
            continue
        P.append(dict(type="rect", x0=c[0], x1=c[1], y0=c[2], ys=c[3], rho=0.0, depth=0.0, through=True, cut=True,
                      precision=1.0, iou=1.0))
    P.sort(key=lambda q: (bbox(q)[2], bbox(q)[0]))
    for pr in P:
        log("  %-6s b %6.2f..%6.2f  a %6.2f..%6.2f  rise %.2f  depth %.2f %s  precision %.2f" % (
            pr["type"], *bbox(pr), pr.get("rho", 0), pr["depth"], "THROUGH" if pr["through"] else "", pr["precision"]))
    back = -A * cameraSide                               # inwards (away from the cameras)

    def P3(q, d=0.0):
        return q[0] * B + q[1] * Av + p["off"] * A + d * back
    objects = [("wall", [np.array([P3(q) for q in poly]) for poly in holedSlab((b0, b1, a0, a1), P)])]
    for k, pr in enumerate(P):
        if pr.get("cut"):
            continue                                     # plain cut: no reveal
        ol = outline(pr)
        d = WALL_THICKNESS if pr["through"] else min(max(pr["depth"], 0.1), WALL_THICKNESS * 1.5)
        sides = []
        for i in range(len(ol)):
            q0, q1 = ol[i], ol[(i + 1) % len(ol)]
            if pr["type"] != "circle" and pr["y0"] <= a0 + 1e-6 and abs(q0[1] - a0) < 1e-6 and abs(q1[1] - a0) < 1e-6:
                continue                                 # no sill under a door
            sides.append(np.array([P3(q0), P3(q0, d), P3(q1, d), P3(q1)]))
        objects.append(("reveal_%02d" % k, sides))
        if not pr["through"]:
            objects.append(("glazing_%02d" % k, [np.array([P3(q, d) for q in ol])]))
    return objects, P, (b0, b1, a0, a1)


def openingsImage(img, M, P, rect):
    """Recess map image with the primitive outlines."""
    s = img.shape[0] / M["med"].shape[0]
    b0, b1, a0, a1 = rect
    for pr in P:
        ol = outline(pr)
        pts = np.stack([(ol[:, 0] - b0) / float(M["cell"]) * s, img.shape[0] - (ol[:, 1] - a0) / float(M["cell"]) * s], 1)
        cv2.polylines(img, [pts.astype(np.int32)], True, (255, 255, 255), 2)
    return img


def holedVolumes(scene, work, large, recesses, stairs, log=print):
    """Large planes + stairs, with each measured wall replaced by its holed slab, reveals and glazing.

    Returns:
        list of quads (dicts with V and name), in the working frame.
    """
    from .recesses import recessImage
    quads = [dict(q) for q in large["quads"]]
    if stairs is not None:
        quads += [dict(V=q["V"], name="stairs_%03d" % k) for k, q in enumerate(stairs["quads"])]
    for pid, M in recesses.items():
        p = planeById(large, pid)
        A = planeAxes(large, p)[0]
        C = scene.centers([f for f in M["frames"]]).mean(0)
        side = np.sign(C @ A - p["off"])
        log(pid)
        objects, P, rect = wallOpenings(scene, pid, large, M, side, log)
        log("  %d primitives" % len(P))
        img = openingsImage(recessImage(M), M, P, rect)
        cv2.imwrite(os.path.join(work, "openings_%s.png" % pid), img)
        tag = "%s_wall_%s" % (p["dom"], planeLabel(p["axis"], p["off"]))
        quads = [q for q in quads if not q["name"].startswith(tag)]
        for name, polys in objects:
            quads += [dict(V=np.array(V), name="%s_%s" % (pid, name)) for V in polys]
    return quads
