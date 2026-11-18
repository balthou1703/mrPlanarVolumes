"""Photometric refinement of plane distances (plane sweep, multi-view NCC).

For a carved plane: orthophotos of the plane shifted by delta along its normal from up to 16 views, local NCC
(11 px window) of each view against a reference view, robust aggregation (best 2/3 of the views), delta = maximum
of the score (coarse then fine, parabolic interpolation), then the tilts (coordinate descent).
"""
import os
import pickle

import cv2
import numpy as np

from .depth import depthFrames

REFINED_FILE = "refined.pkl"
DS = 2


class Refiner:
    """Plane sweep on the key frames of a scene.

    Args:
        scene: Scene.
        work: working folder (key frames).
        masks: semantic masks (people are ignored).
    """

    def __init__(self, scene, work, masks):
        self.scene = scene
        self.masks = masks
        self.candidates = depthFrames(work)
        self._gray = {}

    def gray(self, frame):
        if frame not in self._gray:
            g = cv2.cvtColor(self.scene.image(frame), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            g = cv2.resize(g, (g.shape[1] // DS, g.shape[0] // DS), interpolation=cv2.INTER_AREA)
            labels = self.masks.labels(frame, g.shape)
            if labels is not None:
                g[labels == 1] = np.nan                  # people
            self._gray[frame] = g
        return self._gray[frame]

    def KRt(self, frame):
        i = self.scene.index(frame)
        fx, fy, cx, cy = self.scene.K[i]
        K = np.array([[fx / DS, 0, (cx + .5) / DS - .5], [0, fy / DS, (cy + .5) / DS - .5], [0, 0, 1]])
        return K, self.scene.R[i], self.scene.t[i]

    def homography(self, frame, p, prm, lo, r):
        """prm = (delta, tx, ty): X(u, v) = u b + v a + (off + delta + tx (u - uc) + ty (v - vc)) n (tilted plane)."""
        K, R, t = self.KRt(frame)
        d, tx, ty = prm
        uc, vc = p["_c"]
        bb = p["b"] + tx * p["n"]
        aa = p["a"] + ty * p["n"]
        X0 = (p["off"] + d - tx * uc - ty * vc) * p["n"] + (lo[0] + .5 * r) * bb + (lo[1] + .5 * r) * aa
        return K @ np.stack([R @ bb * r, R @ aa * r, R @ X0 + t], 1)

    def ortho(self, frame, p, prm, lo, r, shape):
        M = self.homography(frame, p, prm, lo, r)
        return cv2.warpPerspective(self.gray(frame), M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

    def chooseViews(self, p, cells, nMax=16):
        """Reference view (best coverage x obliquity / depth) and up to nMax views with 8-60 degree parallax."""
        scored = []
        center = cells.mean(0)
        for n in self.candidates:
            if n < min(p["frames"]) - 30 or n > max(p["frames"]) + 30:
                continue
            K, R, t = self.KRt(n)
            C = -R.T @ t
            Xc = cells @ R.T + t
            z = Xc[:, 2]
            uv = Xc[:, :2] / np.maximum(z[:, None], 1e-6) @ K[:2, :2].T + K[:2, 2]
            h, w = self.gray(n).shape
            inside = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < w) & (uv[:, 1] < h)
            if inside.mean() < 0.25:
                continue
            v = cells[inside] - C
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            cos = np.abs(v @ p["n"]).mean()
            if cos < 0.3:
                continue
            scored.append((n, inside.mean() * cos / np.median(z[inside]), C))
        if len(scored) < 3:
            return None, []
        ref, _, Cr = max(scored, key=lambda s: s[1])
        a1 = (center - Cr) / np.linalg.norm(center - Cr)
        rest = []
        for n, _, C in scored:
            a2 = (center - C) / np.linalg.norm(center - C)
            angle = np.degrees(np.arccos(np.clip(a1 @ a2, -1, 1)))
            if 8 <= angle <= 60:
                rest.append(n)
        if len(rest) < 2:
            return None, []
        idx = np.unique(np.round(np.linspace(0, len(rest) - 1, min(nMax, len(rest)))).astype(int))
        return ref, [rest[i] for i in idx]

    def score(self, p, prm, lo, r, shape, valid, ref, views, returnMap=False):
        R0 = self.ortho(ref, p, prm, lo, r, shape)
        S = []
        for n in views:
            c = ncc(R0, self.ortho(n, p, prm, lo, r, shape))
            c[~valid] = np.nan
            S.append(c)
        S = np.stack(S)
        nv = np.isfinite(S).sum(0)
        Ss = -np.sort(-np.nan_to_num(S, nan=-np.inf), 0)
        k = np.maximum((2 * nv) // 3, 1)                 # best 2/3 of the views (occlusions)
        top = np.where(np.arange(len(S))[:, None, None] < k[None], Ss, 0).sum(0) / k
        ok = nv >= 2
        if returnMap:
            return np.where(ok, top, np.nan)
        return float(top[ok].mean()) if ok.sum() > 200 else np.nan

    def refine(self, o):
        """Refine a carved plane (dict with plane, mask, lo, cell, zmed).

        Returns:
            dict: delta, tx, ty (tilts), n, off, center, peak, base, flat (score without tilt), contrast,
            cellscore (per cell NCC), uc (tilt center), curve, ref, views; None without enough views.
        """
        p = dict(o["plane"])
        m = o["mask"] > 0
        cell = o["cell"]
        ys, xs = np.nonzero(m)
        loC = o["lo"] + np.array([xs.min(), ys.min()]) * cell
        mm = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        iy, ix = np.nonzero(mm)
        p["_c"] = (loC[0] + (ix.mean() + .5) * cell, loC[1] + (iy.mean() + .5) * cell)
        cc = np.stack([iy, ix], 1)[::max(1, len(iy) // 400)]
        cells = np.array([(loC[0] + (j + .5) * cell) * p["b"] + (loC[1] + (i + .5) * cell) * p["a"] + p["off"] * p["n"]
                          for i, j in cc])
        ref, views = self.chooseViews(p, cells)
        if ref is None:
            return None

        def grid(r):
            while (mm.shape[0] * cell / r) * (mm.shape[1] * cell / r) > 2e6:
                r *= 1.4
            shape = (int(np.ceil(mm.shape[0] * cell / r)), int(np.ceil(mm.shape[1] * cell / r)))
            return r, shape, cv2.resize(mm.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        sweep = max(1.0, 0.07 * o["zmed"])
        r, shape, valid = grid(0.05)
        D = np.arange(-sweep, sweep + 1e-6, 0.04)
        S0 = np.array([self.score(p, (d, 0, 0), loC, r, shape, valid, ref, views) for d in D])
        if not np.isfinite(S0).any():
            return None
        d0 = parabolicPeak(D, S0)
        tx = ty = 0.0
        T = np.tan(np.radians(np.arange(-5, 5.01, 1.0)))
        for _ in range(2):                               # tilts (coordinate descent)
            Sx = np.array([self.score(p, (d0, a, ty), loC, r, shape, valid, ref, views) for a in T])
            tx = parabolicPeak(T, Sx)
            Sy = np.array([self.score(p, (d0, tx, a), loC, r, shape, valid, ref, views) for a in T])
            ty = parabolicPeak(T, Sy)
        r, shape, valid = grid(0.025)
        D1 = np.arange(d0 - 0.1, d0 + 0.101, 0.01)
        S1 = np.array([self.score(p, (d, tx, ty), loC, r, shape, valid, ref, views) for d in D1])
        d1 = parabolicPeak(D1, S1)
        sFlat = self.score(p, (d1, 0, 0), loC, r, shape, valid, ref, views)
        tm = self.score(p, (d1, tx, ty), loC, r, shape, valid, ref, views, returnMap=True)
        fin = np.isfinite(tm)
        num = cv2.resize(np.where(fin, tm, 0).astype(np.float32), (mm.shape[1], mm.shape[0]), interpolation=cv2.INTER_AREA)
        den = cv2.resize(fin.astype(np.float32), (mm.shape[1], mm.shape[0]), interpolation=cv2.INTER_AREA)
        cs = np.full(m.shape, np.nan, np.float32)
        cs[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = np.where(den > 0.2, num / np.maximum(den, 1e-6), np.nan)
        nn = p["n"] - tx * p["b"] - ty * p["a"]
        nn /= np.linalg.norm(nn)
        Xc = p["_c"][0] * p["b"] + p["_c"][1] * p["a"] + (p["off"] + d1) * p["n"]
        return dict(delta=d1, tx=float(tx), ty=float(ty), n=nn, off=float(nn @ Xc), center=Xc,
                    peak=float(np.nanmax(S1)), base=float(S0[np.argmin(np.abs(D))]), flat=sFlat,
                    contrast=float(np.nanmax(S0) - np.nanmedian(S0)), cellscore=cs, uc=p["_c"], curve=(D, S0, D1, S1),
                    ref=ref, views=views)


def ncc(A, B, w=11):
    """Local normalized cross-correlation of two images with NaN holes (NaN where untextured or incomplete)."""
    va = np.isfinite(A) & np.isfinite(B)
    A0 = np.where(va, A, 0)
    B0 = np.where(va, B, 0)
    c = cv2.boxFilter(va.astype(np.float32), -1, (w, w))

    def box(x):
        return cv2.boxFilter(x, -1, (w, w))
    c1 = np.maximum(c, 1e-3)
    ma, mb = box(A0) / c1, box(B0) / c1
    vab = box(A0 * B0) / c1 - ma * mb
    vaa = box(A0 * A0) / c1 - ma ** 2
    vbb = box(B0 * B0) / c1 - mb ** 2
    r = vab / np.sqrt(np.maximum(vaa * vbb, 1e-10))
    r[(c < 0.8) | (vaa < 2e-4) | (vbb < 2e-4)] = np.nan
    return r


def parabolicPeak(xs, scores):
    """Position of the maximum of sampled scores, refined by a parabola through the 3 best samples."""
    i = int(np.nanargmax(scores))
    best = xs[i]
    if 0 < i < len(scores) - 1 and np.isfinite(scores[i - 1:i + 2]).all():
        a, b, c = scores[i - 1:i + 2]
        den = a - 2 * b + c
        if den < 0:
            best = xs[i] + 0.5 * (a - c) / den * (xs[1] - xs[0])
    return float(best)


def refineDomain(refiner, domainDir, log=print):
    """Refine all carved planes of a domain -> <domainDir>/refined.pkl."""
    with open(os.path.join(domainDir, "carve.pkl"), "rb") as f:
        carved = pickle.load(f)
    out = []
    for i, o in enumerate(carved):
        if o is None or o["mask"].sum() < 250:
            out.append(None)
            continue
        rr = refiner.refine(o)
        out.append(rr)
        if rr:
            log("plane %2d fam %2d off %7.2f -> delta %+6.3f tilt %+.1f/%+.1f deg, NCC %.2f -> %.2f (untilted %.2f), "
                "contrast %.2f, %d views" % (i, o["plane"]["fam"], o["plane"]["off"], rr["delta"],
                                             np.degrees(np.arctan(rr["tx"])), np.degrees(np.arctan(rr["ty"])),
                                             rr["base"], rr["peak"], rr["flat"], rr["contrast"], len(rr["views"])))
        else:
            log("plane %2d: not enough views" % i)
    with open(os.path.join(domainDir, REFINED_FILE), "wb") as f:
        pickle.dump(out, f)
    return out
