"""Planar segments per key frame: regions of the monocular normal map close to a normal family, split where the
plane distance jumps, each anchored on the track points it contains (or on the scaled monocular depth otherwise).
"""
import os
import pickle

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from .depth import depthFrames, loadDepth, trackScale

SEGMENTS_FILE = "segments.pkl"
DS = 4               # map subsampling
MIN_PIXELS = 150     # smallest segment (subsampled pixels)


def frameSegments(scene, work, frame, families, masks):
    """Planar segments of one key frame.

    Returns:
        list of dicts: frame, fam (family index), off (n.X), off_moge, src ('trk' anchored on the track or 'moge'),
        ntrk, area (pixels), depth, pts (3D sample on the plane), bbox.
    """
    depth, normal, valid = loadDepth(work, frame)
    i = scene.index(frame)
    R, t = scene.R[i], scene.t[i]
    fx, fy, cx, cy = scene.K[i]
    H, W = depth.shape
    xy, XwTrack = scene.trackObservations(frame)
    (a, b), zTrack, _ = trackScale(scene, frame, depth, valid)
    h, w = H // DS, W // DS
    d = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST) * a + b
    nr = cv2.resize(normal, (w, h), interpolation=cv2.INTER_AREA)
    nr /= np.linalg.norm(nr, axis=-1, keepdims=True) + 1e-9
    mk = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    mk &= np.isfinite(d) & (d > 0)
    vv, uu = np.mgrid[0:h, 0:w]
    uc = (uu + 0.5) * DS - 0.5
    vc = (vv + 0.5) * DS - 0.5
    Pc = np.stack([(uc - cx) / fx * d, (vc - cy) / fy * d, d], -1)
    Xw = (Pc - t) @ R
    Nw = nr @ R
    # smoothness: normals consistent over 5x5 (foliage removed)
    smooth = np.linalg.norm(np.stack([uniform_filter(nr[..., k], 5) for k in range(3)], -1), axis=-1)
    mk &= smooth > 0.95
    cosFam = Nw @ families.T
    lab = np.argmax(cosFam, -1)
    best = np.max(cosFam, -1)
    mk &= best > np.cos(np.radians(12))
    dmax = np.percentile(zTrack, 95) * 1.5
    mk &= d < dmax
    labels = masks.labels(frame, (h, w))
    if labels is not None:                             # people are never planes
        mk &= cv2.dilate((labels == 1).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
    segs = []
    tu = np.clip((xy[:, 0] / DS).astype(int), 0, w - 1)
    tv = np.clip((xy[:, 1] / DS).astype(int), 0, h - 1)
    for f in range(len(families)):
        m = (mk & (lab == f)).astype(np.uint8)
        if m.sum() < MIN_PIXELS:
            continue
        off = Xw @ families[f]                        # signed distance along the normal
        # connected components with continuous distance: cut where the distance gradient is strong (3 % of depth)
        gx = np.abs(np.diff(off, axis=1, append=off[:, -1:]))
        gy = np.abs(np.diff(off, axis=0, append=off[-1:]))
        step = 0.03 * d
        m &= ((gx < step) & (gy < step)).astype(np.uint8)
        nc, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        for k in range(1, nc):
            if st[k, cv2.CC_STAT_AREA] < MIN_PIXELS:
                continue
            ck = cc == k
            tin = ck[tv, tu]
            oMoge = np.median(off[ck])
            src, o = "moge", oMoge
            if tin.sum() >= 8:
                ot = XwTrack[tin] @ families[f]
                near = np.abs(ot - oMoge) < 0.15 * np.median(d[ck])    # track points on the plane (not foreground)
                if near.sum() >= 6:
                    o = np.median(ot[near])
                    src = "trk"
            pts = Xw[ck]
            if len(pts) > 400:
                pts = pts[np.random.default_rng(k).choice(len(pts), 400, replace=False)]
            pts = pts + (o - oMoge) * families[f]        # sample moved onto the retained plane
            segs.append(dict(frame=frame, fam=f, off=o, off_moge=oMoge, src=src, ntrk=int(tin.sum()),
                             area=int(ck.sum()), depth=float(np.median(d[ck])), pts=pts.astype(np.float32),
                             bbox=(st[k, 0], st[k, 1], st[k, 2], st[k, 3])))
    return segs


def computeSegments(scene, work, families, masks, log=print):
    """Segments of all key frames -> <work>/segments.pkl."""
    out = []
    for frame in depthFrames(work):
        out += frameSegments(scene, work, frame, families, masks)
    with open(os.path.join(work, SEGMENTS_FILE), "wb") as f:
        pickle.dump(out, f)
    src = np.array([s["src"] for s in out])
    log("planar segments: %d, anchored on the track %.0f %%" % (len(out), 100 * (src == "trk").mean() if len(out) else 0))
    dm = np.array([abs(s["off"] - s["off_moge"]) / s["depth"] for s in out if s["src"] == "trk"])
    if len(dm):
        log("monocular / track distance gap relative to depth: median %.3f, p90 %.3f" % (np.median(dm), np.percentile(dm, 90)))
    return out


def loadSegments(work):
    with open(os.path.join(work, SEGMENTS_FILE), "rb") as f:
        return pickle.load(f)
