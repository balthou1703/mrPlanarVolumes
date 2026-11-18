"""Straight stairs from the monocular normals: detection in the images, per-frame comb fit, global stairs.

1. detection: vertical alternation of "tread" (normal up) and "riser" (horizontal normal) bands in a key frame;
2. per frame: comb of heights (treads) and of positions along the climb (risers), Rayleigh phase concentration;
3. global: weighted median of the steps over the reliable frames, phases on all the points -> one quad per riser
   and per tread. The dense mesh step (denseStructure) refines it into individual steps when available.
"""
import os
import pickle

import cv2
import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d

from .depth import alignedGeometry, depthFrames, loadDepth
from .geometry import comb, weightedMedian, writeObj

STAIRS_FILE = "stairs.pkl"
DS = 2
MAX_DISTANCE = 20.0      # frames farther than this from the stairs are not used
MIN_FRAMES = 3           # stairs seen clearly in fewer frames are not reliable


def stairMap(scene, work, frame, up):
    """Tread / riser labels (1 / -1) and a stairs score per pixel of a key frame, at 1/DS resolution."""
    _, normal, _ = loadDepth(work, frame)
    R = scene.R[scene.index(frame)]
    H, W = normal.shape[:2]
    nr = cv2.resize(normal, (W // DS, H // DS), interpolation=cv2.INTER_AREA)
    Nw = nr @ R
    Nw /= np.linalg.norm(Nw, axis=-1, keepdims=True) + 1e-9
    c = Nw @ up
    tread = c > np.cos(np.radians(20))
    riser = np.abs(c) < np.sin(np.radians(20))
    labels = np.where(tread, 1, np.where(riser, -1, 0)).astype(np.float32)
    # number of distinct tread bands (band starts) in a vertical window, with risers present
    tr = cv2.morphologyEx(tread.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 9), np.uint8)) > 0
    start = np.zeros_like(tr)
    start[1:] = tr[1:] & ~tr[:-1]
    win = 121
    count = uniform_filter1d(start.astype(np.float32), win, axis=0) * win
    ok = uniform_filter(riser.astype(np.float32), 31) > 0.2
    return labels, uniform_filter(count * ok, 21)


def stairsUp(families):
    """Family closest to the image up direction (-y), oriented up (the stairs detection convention)."""
    ref = np.array([0, -1.0, 0])
    up = families[np.argmax(np.abs(families @ ref))]
    return up * np.sign(up @ ref)


def fitFrames(scene, work, families, frameMin, frameMax, log=print):
    """Straight stairs fitted in each key frame of [frameMin, frameMax] where stairs are detected."""
    up = stairsUp(families)
    res = []
    for n in depthFrames(work):
        if not frameMin <= n <= frameMax:
            continue
        _, score = stairMap(scene, work, n, up)
        m = (score > 3).astype(np.uint8)
        if m.sum() < 500:
            continue
        m = cv2.dilate(m, np.ones((31, 31), np.uint8)) > 0
        H2, W2 = m.shape
        full = cv2.resize(m.astype(np.uint8), (W2 * 2, H2 * 2), interpolation=cv2.INTER_NEAREST) > 0
        g = alignedGeometry(scene, work, n, ds=2, mask=cv2.dilate(full.astype(np.uint8), np.ones((101, 101), np.uint8)) > 0)
        X, N = g["X"], g["N"]
        ok = m & np.isfinite(g["dd"])
        cth = N @ up
        tr = ok & (cth > np.cos(np.radians(15)))
        rs = ok & (np.abs(cth) < np.sin(np.radians(15)))
        if tr.sum() < 200 or rs.sum() < 200:
            continue
        # climbing direction: opposite to the mean horizontal normal of the risers (facing the camera)
        nr = N[rs]
        nr = nr - np.outer(nr @ up, up)
        nr /= np.linalg.norm(nr, axis=1, keepdims=True)
        direction = -nr.mean(0)
        direction /= np.linalg.norm(direction)
        side = np.cross(up, direction)
        ht = X[tr] @ up
        sr = X[rs] @ direction
        dist = np.median(g["dd"][ok])
        rh, sh, h0 = comb(ht, np.ones_like(ht), 0.05 * 1.6, 0.35 * 1.6)
        rd, sd, d0 = comb(sr, np.ones_like(sr), 0.1 * 1.6, 0.7 * 1.6)
        ks = np.round((ht - h0) / sh)
        ku, cnt = np.unique(ks, return_counts=True)
        steps = ku[cnt > 0.02 * len(ht)]
        lat = X[tr] @ side
        res.append(dict(frame=n, dist=dist, up=up, dir=direction, side=side, step_h=sh, h0=h0, Rh=rh, step_d=sd, d0=d0,
                        Rd=rd, nsteps=len(steps), kmin=steps.min() if len(steps) else 0, kmax=steps.max() if len(steps) else 0,
                        lat=(np.percentile(lat, 5), np.percentile(lat, 95)), tread_pts=X[tr][::7], riser_pts=X[rs][::7]))
        log("  frame %d: distance %.1f, rise %.3f (R %.2f), run %.3f (R %.2f), %d steps" % (
            n, dist, sh, rh, sd, rd, len(steps)))
    return res


def buildStairs(fits, work, log=print):
    """Global straight stairs from the per-frame fits -> <work>/stairs.pkl, stairs.obj (removed when not found).

    Returns:
        dict or None.
    """
    S = [f for f in fits if f["dist"] < MAX_DISTANCE and f["Rh"] > 0.9 and f["Rd"] > 0.5 and f["nsteps"] >= 3]
    log("stairs: %d reliable frames %s" % (len(S), [f["frame"] for f in S]))
    if len(S) < MIN_FRAMES:
        for name in (STAIRS_FILE, "stairs.obj"):
            if os.path.exists(os.path.join(work, name)):
                os.remove(os.path.join(work, name))
        log("stairs: fewer than %d reliable frames, no stairs" % MIN_FRAMES)
        return None
    w = np.array([f["Rh"] * f["Rd"] / f["dist"] for f in S])
    up = S[0]["up"]
    direction = np.sum([wi * f["dir"] for wi, f in zip(w, S)], 0)
    direction -= (direction @ up) * up
    direction /= np.linalg.norm(direction)
    side = np.cross(up, direction)
    sh = weightedMedian(np.array([f["step_h"] for f in S]), w)
    sd = weightedMedian(np.array([f["step_d"] for f in S]), w)
    log("stairs: rise %.3f u, run %.3f u, direction %s" % (sh, sd, np.round(direction, 3)))
    T = np.concatenate([f["tread_pts"] for f in S])
    Rr = np.concatenate([f["riser_pts"] for f in S])
    wt = np.concatenate([np.full(len(f["tread_pts"]), wi) for f, wi in zip(S, w)])
    wr = np.concatenate([np.full(len(f["riser_pts"]), wi) for f, wi in zip(S, w)])
    # common lateral band (treads and risers detected outside of the stairs are dropped)
    latT = T @ side
    lc = weightedMedian(latT, wt)
    half = 0.5 * weightedMedian(np.array([f["lat"][1] - f["lat"][0] for f in S]), w)
    kt = np.abs(latT - lc) < half * 1.2
    kr = np.abs(Rr @ side - lc) < half * 1.2
    T, wt, Rr, wr = T[kt], wt[kt], Rr[kr], wr[kr]
    ht = T @ up
    sr = Rr @ direction
    h0 = (np.angle(np.sum(wt * np.exp(2j * np.pi * ht / sh))) / (2 * np.pi) * sh) % sh
    d0 = (np.angle(np.sum(wr * np.exp(2j * np.pi * sr / sd))) / (2 * np.pi) * sd) % sd
    rh = np.abs(np.sum(wt * np.exp(2j * np.pi * ht / sh))) / wt.sum()
    rd = np.abs(np.sum(wr * np.exp(2j * np.pi * sr / sd))) / wr.sum()
    log("stairs: global phase concentration, treads %.2f, risers %.2f" % (rh, rd))
    kh = np.round((ht - h0) / sh).astype(int)
    ks = np.floor((T @ direction - d0) / sd).astype(int)       # riser interval containing each tread point
    off = np.bincount((ks - kh) - (ks - kh).min(), weights=wt).argmax() + (ks - kh).min()
    ku, kc = np.unique(kh, return_counts=True)
    steps = ku[kc > 0.03 * len(kh)]
    k0, k1 = steps.min(), steps.max()
    log("stairs: steps %d to %d (%d steps)" % (k0, k1, k1 - k0 + 1))
    L0, L1 = lc - half, lc + half

    def pt(s, h, lateral):
        return s * direction + h * up + lateral * side
    quads = []
    for k in range(k0, k1 + 1):
        h = h0 + k * sh
        hb = h - sh
        s0 = d0 + (k + off) * sd
        s1 = s0 + sd
        quads.append(np.array([pt(s0, hb, L1), pt(s0, hb, L0), pt(s0, h, L0), pt(s0, h, L1)]))   # riser
        quads.append(np.array([pt(s0, h, L0), pt(s0, h, L1), pt(s1, h, L1), pt(s1, h, L0)]))     # tread
    stairs = dict(quads=[dict(V=V, fam=2 if i % 2 else 5, plane=0, off=0, area=1, frames=[], ntrk=1)
                         for i, V in enumerate(quads)],
                  up=up, params=dict(step_h=sh, step_d=sd, h0=h0, d0=d0, dir=direction, side=side, k0=int(k0), k1=int(k1),
                                     width=2 * half, off=int(off)))
    with open(os.path.join(work, STAIRS_FILE), "wb") as f:
        pickle.dump(stairs, f)
    writeObj(os.path.join(work, "stairs.obj"), [("stairs", quads)])
    log("stairs: width %.2f u, %d quads" % (2 * half, len(quads)))
    return stairs


def loadStairs(work):
    path = os.path.join(work, STAIRS_FILE)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
