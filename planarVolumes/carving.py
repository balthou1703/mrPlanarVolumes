"""Extent of the planes by multi-view free-space voting.

For each cell of a plane and each key frame where it projects:
  monocular depth on the plane and consistent normal -> +1 (surface);
  monocular depth clearly beyond (or sky)           -> +1 free (seen through: opening or nonexistent plane);
  object in front                                   -> nothing (occlusion).
"""
import os
import pickle

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from .depth import depthFrames, loadDepth, trackScale
from .geometry import maxRect, writeObj

CARVE_FILE = "carve.pkl"
CARVED_FILE = "carved.pkl"
DS = 4
CELL = 0.06
MARGIN = 1.0             # search area around the plane points (u)
MAX_CELLS = 600_000


def loadFrame(scene, work, frame, masks, useWindows):
    """Key frame data for the votes, at 1/DS resolution: scaled depth (inf = sky/invalid), world normals,
    vegetation, people and window masks."""
    depth, normal, valid = loadDepth(work, frame)
    i = scene.index(frame)
    R, t = scene.R[i], scene.t[i]
    H, W = depth.shape
    (a, b), zTrack, _ = trackScale(scene, frame, depth, valid)
    h, w = H // DS, W // DS
    dd = cv2.resize(np.nan_to_num(depth, nan=np.inf, posinf=np.inf), (w, h), interpolation=cv2.INTER_NEAREST) * a + b
    mk = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    dd[~mk] = np.inf
    nr = cv2.resize(normal, (w, h), interpolation=cv2.INTER_AREA)
    nr /= np.linalg.norm(nr, axis=-1, keepdims=True) + 1e-9
    fx, fy, cx, cy = scene.K[i]
    dmax = np.percentile(zTrack, 95) * 1.5
    # vegetation: locally inconsistent normals or saturated green
    coherence = np.linalg.norm(np.stack([uniform_filter(nr[..., k], 5) for k in range(3)], -1), axis=-1)
    hsv = cv2.cvtColor(cv2.resize(scene.image(frame), (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 60)
    vegetation = (coherence < 0.93) | green
    people = np.zeros((h, w), bool)
    windows = np.zeros((h, w), bool)
    labels = masks.labels(frame, (h, w))
    if labels is not None:
        people = cv2.dilate((labels == 1).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        windows = labels == 2
    if not useWindows:
        windows[:] = False
    return dict(veg=vegetation, pers=people, win=windows, R=R, t=t,
                K=(fx / DS, fy / DS, (cx + 0.5) / DS - 0.5, (cy + 0.5) / DS - 0.5),
                dd=dd.astype(np.float32), N=(nr @ R).astype(np.float16), dmax=dmax)


def labelFrames(planes, frameData):
    """Exclusive label per pixel: the candidate plane that best explains depth and normal (-1 none, -2 vegetation or
    person: no vote, -3 window pane: seen through)."""
    normals = np.array([p["n"] for p in planes])
    offsets = np.array([p["off"] for p in planes])
    for f in frameData.values():
        h, w = f["dd"].shape
        fx, fy, cx, cy = f["K"]
        v, u = np.mgrid[0:h, 0:w]
        dc = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones((h, w))], -1).reshape(-1, 3)
        dw = dc @ f["R"]
        cc = -f["R"].T @ f["t"]
        t = (offsets[None] - (normals @ cc)[None]) / (dw @ normals.T)
        dd = f["dd"].reshape(-1, 1)
        res = np.abs(dd / np.where(t > 0, t, np.nan) - 1)
        nok = np.abs(f["N"].reshape(-1, 3).astype(np.float32) @ normals.T) > np.cos(np.radians(12))
        res = np.where(nok & np.isfinite(res), res, np.inf)
        lab = np.argmin(res, 1)
        best = res[np.arange(len(res)), lab]
        lab[best > 0.04] = -1
        lab[f["veg"].reshape(-1)] = -2
        lab[f["pers"].reshape(-1)] = -2
        lab[f["win"].reshape(-1)] = -3
        f["lab"] = lab.reshape(h, w)


def carvePlanes(planes, frameData, log=print):
    """Surface (S) and free (F) votes per cell of each plane, and the resulting surface mask."""
    labelFrames(planes, frameData)
    out = []
    for pi, p in enumerate(planes):
        n, a, b, off = p["n"], p["a"], p["b"], p["off"]
        lo = np.floor((p["lo"] - MARGIN) / CELL) * CELL
        hi = p["hi"] + MARGIN
        nx, ny = np.ceil((hi - lo) / CELL).astype(int)
        if nx * ny > MAX_CELLS:
            log("plane %d too large (%d x %d cells): skipped" % (pi, nx, ny))
            out.append(None)
            continue
        gu, gv = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * CELL, lo[1] + (np.arange(ny) + 0.5) * CELL)
        Xg = (gu[..., None] * b + gv[..., None] * a + off * n).reshape(-1, 3)
        S = np.zeros(len(Xg), np.float32)
        Fv = np.zeros(len(Xg), np.float32)
        depths = []
        for f in frameData.values():
            Xc = Xg @ f["R"].T + f["t"]
            z = Xc[:, 2]
            cc = -f["R"].T @ f["t"]
            fx, fy, cx, cy = f["K"]
            ok = (z > 0.3) & (z < f["dmax"])
            u = np.round(Xc[:, 0] / np.maximum(z, 1e-6) * fx + cx).astype(int)
            v = np.round(Xc[:, 1] / np.maximum(z, 1e-6) * fy + cy).astype(int)
            h, w = f["dd"].shape
            ok &= (u >= 0) & (v >= 0) & (u < w) & (v < h)
            if ok.sum() < 20:
                continue
            idx = np.where(ok)[0]
            r = f["dd"][v[idx], u[idx]] / z[idx]
            view = Xg[idx] - cc
            view /= np.linalg.norm(view, axis=1, keepdims=True)
            notGrazing = np.abs(view @ n) > 0.25
            li = f["lab"][v[idx], u[idx]]
            agree = (li == pi) & notGrazing
            free = ((r > 1.15) | (li == -3)) & notGrazing & (li != -2)
            S[idx] += agree
            Fv[idx] += free
            if agree.any():
                depths.append(np.median(z[idx][agree]))
        S = S.reshape(ny, nx)
        Fv = Fv.reshape(ny, nx)
        m = ((S >= 3) & (S > 1.5 * Fv)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        out.append(dict(plane=p, S=S, F=Fv, mask=m, lo=lo, cell=CELL, zmed=float(np.median(depths)) if depths else np.inf))
    return out


def mergeOverlapping(planes, out):
    """Merge parallel planes closer than 4 % of the viewing distance whose footprints overlap."""
    parent = list(range(len(planes)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    cells = []
    for o in out:
        if o is None or o["mask"].sum() == 0:
            cells.append(set())
            continue
        ys, xs = np.nonzero(o["mask"])
        gi = np.round(o["lo"][0] / CELL).astype(int) + xs
        gj = np.round(o["lo"][1] / CELL).astype(int) + ys
        cells.append(set(zip(gi.tolist(), gj.tolist())))
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            if planes[i]["fam"] != planes[j]["fam"] or not cells[i] or not cells[j]:
                continue
            zm = min(out[i]["zmed"], out[j]["zmed"])
            if abs(planes[i]["off"] - planes[j]["off"]) > 0.04 * zm:
                continue
            if len(cells[i] & cells[j]) > 0.2 * min(len(cells[i]), len(cells[j])):
                parent[root(j)] = root(i)
    groups = {}
    for i in range(len(planes)):
        if out[i] is None or out[i]["mask"].sum() == 0:
            continue
        groups.setdefault(root(i), []).append(i)
    merged = []
    for g in groups.values():
        w = np.array([out[i]["S"][out[i]["mask"] > 0].sum() for i in g])
        p = dict(planes[g[0]])
        p["off"] = float(np.average([planes[i]["off"] for i in g], weights=w))
        p["lo"] = np.min([planes[i]["lo"] for i in g], 0)
        p["hi"] = np.max([planes[i]["hi"] for i in g], 0)
        p["frames"] = sorted(set().union(*[planes[i]["frames"] for i in g]))
        p["ntrk"] = sum(planes[i]["ntrk"] for i in g)
        p["merged"] = len(g)
        merged.append(p)
    return merged


def carveDomain(scene, work, domainDir, frameMin, frameMax, masks, interior, log=print):
    """Carve the candidate planes of a domain -> <domainDir>/carve.pkl (votes and masks), carved.pkl, carved.obj.

    Args:
        interior: indoor domain: window panes are openings (outdoors, windows are part of the facade).
    """
    with open(os.path.join(domainDir, "planes.pkl"), "rb") as f:
        d = pickle.load(f)
    planes = [p for p in d["planes"] if "a" in p and len(p["frames"]) >= 4 and (p["ntrk"] >= 2 or p.get("src") == "moge")]
    # nearly coplanar planes (same family, close distances): union of the extents
    planes.sort(key=lambda p: (p["fam"], p["off"]))
    merged = []
    for p in planes:
        q = merged[-1] if merged else None
        if q is not None and q["fam"] == p["fam"] and abs(p["off"] - q["off"]) < 0.08 and q.get("src") == p.get("src"):
            w0, w1 = q["ntrk"], p["ntrk"]
            q["off"] = (q["off"] * w0 + p["off"] * w1) / (w0 + w1)
            q["lo"] = np.minimum(q["lo"], p["lo"])
            q["hi"] = np.maximum(q["hi"], p["hi"])
            q["frames"] = sorted(set(q["frames"]) | set(p["frames"]))
            q["ntrk"] += p["ntrk"]
        else:
            merged.append(dict(p))
    planes = merged
    log("%s: %d candidate planes after merging" % (os.path.basename(domainDir), len(planes)))
    frames = [n for n in depthFrames(work) if frameMin <= n <= frameMax]
    frameData = {n: loadFrame(scene, work, n, masks, interior) for n in frames}
    log("  %d key frames loaded" % len(frameData))
    out = carvePlanes(planes, frameData, log)
    for it in range(2):
        merged = mergeOverlapping(planes, out)
        log("  merge %d: %d -> %d planes" % (it, len(planes), len(merged)))
        if len(merged) == len(planes):
            break
        planes = merged
        out = carvePlanes(planes, frameData, log)
    quads = []
    for pi, o in enumerate(out):
        if o is None:
            continue
        p = o["plane"]
        n, a, b, off = p["n"], p["a"], p["b"], p["off"]
        lo = o["lo"]
        mm = o["mask"].copy()
        for _ in range(25):
            area, x0, y0, x1, y1 = maxRect(mm)
            if area * CELL ** 2 < 0.3:
                break
            mm[y0:y1, x0:x1] = 0
            u0, v0 = lo + np.array([x0, y0]) * CELL
            u1, v1 = lo + np.array([x1, y1]) * CELL
            corners = [uu * b + vv * a + off * n for uu, vv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
            quads.append(dict(V=np.array(corners), fam=p["fam"], plane=pi, off=off, area=area * CELL ** 2,
                              frames=p["frames"], ntrk=p["ntrk"]))
    with open(os.path.join(domainDir, CARVE_FILE), "wb") as f:
        pickle.dump(out, f)
    with open(os.path.join(domainDir, CARVED_FILE), "wb") as f:
        pickle.dump(dict(planes=planes, quads=quads, up=d["up"]), f)
    writeObj(os.path.join(domainDir, "carved.obj"), [("plane%03d_q%03d_fam%02d" % (q["plane"], i, q["fam"]), [q["V"]])
                                                     for i, q in enumerate(quads)])
    log("  %d rectangles" % len(quads))
    return out
