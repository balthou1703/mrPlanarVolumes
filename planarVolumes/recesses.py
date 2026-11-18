"""Recess map of a large wall: for each cell of the plane, depth of the real surface BEHIND the plane, measured on
all the key frames that see it.

- monocular depth rescaled per frame on the plane itself (scale = mode of the ratio monocular / plane depth, plus a
  low-frequency correction of the monocular bias);
- samples in front of the plane (open shutters, chairs, vegetation) and people: ignored (occlusions);
- median per cell -> d > 0: recess (reveal, recessed glazing), large d: opening (open door).
"""
import os

import cv2
import numpy as np

from .depth import depthFrames, loadDepth
from .largePlanes import INPLANE

CELL = 0.05


def planeId(p):
    """Identifier of a large plane, e.g. 'exterior_F1.51'."""
    return "%s_%s%.2f" % (p["dom"], p["axis"], p["off"])


def planeById(large, pid):
    for p in large["planes"]:
        if pid == planeId(p):
            return p
    raise KeyError(pid)


def planeAxes(large, p):
    """(normal, b, a) of a large plane."""
    F = large["frame"]
    b, a = INPLANE[p["axis"]]
    return F[p["axis"]], F[b], F[a]


def measuredWalls(large, minSupport=8.0):
    """Identifiers of the measured large walls (not the floors nor the planes added by rules)."""
    return [planeId(p) for p in large["planes"] if p["axis"] != "U" and not p.get("name") and p["support"] >= minSupport]


def recessMap(scene, work, masks, p, large, frameMin, frameMax, margin=0.0):
    """Recess depths behind a large plane.

    Returns:
        dict: med (rows from the bottom, cells of CELL), n (samples per cell), occ (occlusions per cell),
        rect (b0, b1, a0, a1), cell, frames (used), off, axis; None with fewer than 3 usable frames.
    """
    A, B, Av = planeAxes(large, p)
    b0, b1, a0, a1 = p["rect"]
    b0 -= margin
    b1 += margin
    a1 += margin
    nb, na = int(np.ceil((b1 - b0) / CELL)), int(np.ceil((a1 - a0) / CELL))
    BB, AA = np.meshgrid(b0 + (np.arange(nb) + .5) * CELL, a0 + (np.arange(na) + .5) * CELL)
    X = BB.reshape(-1, 1) * B + AA.reshape(-1, 1) * Av + p["off"] * A
    D, used = [], []
    for n in depthFrames(work):
        if not frameMin <= n <= frameMax:
            continue
        uv, zp = scene.project(n, X)
        depth, _, valid = loadDepth(work, n)
        H, W = depth.shape
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        inside = (zp > 0.3) & (u >= 0) & (v >= 0) & (u < W) & (v < H)
        if inside.mean() < 0.02:
            continue
        C = scene.center(n)
        ray = X - C
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        ok = inside.copy()
        ok[inside] &= valid[v[inside], u[inside]]
        labels = masks.labels(n, (H, W))
        if labels is not None:
            ok[ok] &= labels[v[ok], u[ok]] != 1
        ok &= np.abs(ray @ A) > 0.3
        if ok.sum() < 500:
            continue
        r = np.full(len(X), np.nan, np.float32)
        r[ok] = depth[v[ok], u[ok]] / zp[ok]
        # frame scale: mode of the ratio (most of the plane is on the plane)
        rr = r[ok]
        rr = rr[np.isfinite(rr)]
        h, e = np.histogram(np.log(rr), bins=np.arange(-3.0, 3.0, 0.005))
        s = np.exp(e[np.argmax(cv2.GaussianBlur(h[None].astype(np.float32), (0, 0), 2).ravel())] + 0.0025)
        # low-frequency monocular bias (grazing views, from below): log(r) ~ c0 + c1 u + c2 v, robust, on the plane
        lr = np.log(np.where(ok & np.isfinite(r) & (r > 0), r, np.nan))
        kk = np.isfinite(lr)
        un, vn = uv[:, 0] / W - .5, uv[:, 1] / H - .5
        Mx = np.stack([np.ones(kk.sum()), un[kk], vn[kk]], 1)
        y = lr[kk]
        c = np.array([np.log(s), 0.0, 0.0])
        for _ in range(8):
            w = 1 / (1 + ((y - Mx @ c) / 0.015) ** 2)    # Cauchy: recesses and protrusions do not count
            c = np.linalg.lstsq(Mx * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        sc = np.exp(c[0] + c[1] * un + c[2] * vn)
        side = np.sign(C @ A - p["off"])
        Xm = C + (X - C) * (r / sc)[:, None]
        D.append(((p["off"] - Xm @ A) * side).astype(np.float16))    # > 0: behind the plane
        used.append(n)
    if len(D) < 3:
        return None
    D = np.stack(D).astype(np.float32)                    # (frames, cells)
    occ = D < -0.25                                       # in front: occlusion
    Dv = np.where(occ, np.nan, D)
    nval = np.isfinite(Dv).sum(0)
    med = np.full(D.shape[1], np.nan, np.float32)
    k = nval >= 3
    med[k] = np.nanmedian(Dv[:, k], 0)
    return dict(med=med.reshape(na, nb), n=nval.reshape(na, nb), occ=occ.sum(0).reshape(na, nb),
                rect=(b0, b1, a0, a1), cell=CELL, frames=used, off=p["off"], axis=p["axis"])


def recessImage(M):
    """Color view of a recess map (turbo, -0.3 to 1 u), top up."""
    med = M["med"]
    img = np.zeros(med.shape + (3,), np.uint8)
    v = np.clip(med / 1.0, -0.3, 1.0)
    fin = np.isfinite(med)
    col = cv2.applyColorMap(((np.nan_to_num(v) + 0.3) / 1.3 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[fin] = col[fin]
    img = img[::-1]
    return cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)


def computeRecesses(scene, work, masks, large, log=print):
    """Recess maps of all the measured walls -> <work>/recess_<id>.npz and a control image recess_<id>.png.

    Returns:
        dict {plane id: map}.
    """
    out = {}
    for pid in measuredWalls(large):
        p = planeById(large, pid)
        fr = p.get("frames") or []
        fmin, fmax = (min(fr) - 30, max(fr) + 30) if fr else (-10 ** 9, 10 ** 9)
        M = recessMap(scene, work, masks, p, large, fmin, fmax)
        if M is None:
            log("%s: not enough frames" % pid)
            continue
        np.savez_compressed(os.path.join(work, "recess_%s.npz" % pid), **{k: np.asarray(v) for k, v in M.items()})
        cv2.imwrite(os.path.join(work, "recess_%s.png" % pid), recessImage(M))
        m = M["med"]
        log("%s: %d frames, measured cells %.0f %%, recess > 0.25 u: %.1f %%" % (
            pid, len(M["frames"]), 100 * np.isfinite(m).mean(), 100 * np.nanmean(m > 0.25)))
        out[pid] = M
    return out
