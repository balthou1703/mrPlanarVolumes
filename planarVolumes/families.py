"""Normal families: dominant orientations of the surfaces of the shot, from the monocular normals of the key frames.

Greedy mode search on the sphere (density in a 6 degree cone, refined mean, removal within 10 degrees), weighted by
the seen area (depth squared). Opposite normals are the same family; floors and ceilings are oriented up.
"""
import os

import numpy as np

from .depth import depthFrames, loadDepth

FAMILIES_FILE = "families.npy"


def estimateFamilies(scene, work, log=print):
    """Dominant normal directions, by decreasing importance -> <work>/families.npy.

    Returns:
        (k, 3) array of unit normals.
    """
    frames = depthFrames(work)
    rng = np.random.default_rng(0)
    normals, weights = [], []
    for frame in frames[::max(1, len(frames) // 120)]:
        depth, normal, valid = loadDepth(work, frame)
        nr, d, m = normal[::8, ::8], depth[::8, ::8], valid[::8, ::8]
        R = scene.R[scene.index(frame)]
        ok = m & np.isfinite(d) & np.isfinite(nr).all(-1)
        N = nr[ok] @ R                                     # camera -> world
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
        normals.append(N)
        weights.append((d[ok] / np.median(d[ok])) ** 2)
    N = np.concatenate(normals)
    W = np.concatenate(weights)
    if len(N) > 600000:
        k = rng.choice(len(N), 600000, replace=False)
        N, W = N[k], W[k]
    log("normal families: %d normals" % len(N))
    families, support = [], []
    left = np.ones(len(N), bool)
    cos6, cos10 = np.cos(np.radians(6)), np.cos(np.radians(10))
    for _ in range(24):
        idx = np.nonzero(left)[0]
        if len(idx) < 100:
            break
        cand = N[rng.choice(idx, min(400, len(idx)), replace=False)]
        density = np.array([(W[idx] * (np.abs(N[idx] @ c) > cos6)).sum() for c in cand])
        c = cand[np.argmax(density)]
        for _ in range(5):                                 # mean in the cone, signs aligned
            k = idx[np.abs(N[idx] @ c) > cos6]
            s = np.sign(N[k] @ c)[:, None]
            c = (N[k] * s * W[k, None]).sum(0)
            c /= np.linalg.norm(c)
        w = W[idx][np.abs(N[idx] @ c) > cos6].sum()
        if w < 0.004 * W.sum():
            break
        families.append(c)
        support.append(w)
        left &= np.abs(N @ c) < cos10
    families = np.array(families)
    up = scene.meanUp()
    for i, c in enumerate(families):
        if abs(c @ up) > 0.7 and c @ up < 0:              # floors and ceilings: towards the camera up
            families[i] = -c
    np.save(os.path.join(work, FAMILIES_FILE), families.astype(np.float64))
    for c, w in zip(families, support):
        log("  %s %.1f %%%s" % (np.round(c, 3), 100 * w / W.sum(), " (vertical)" if abs(c @ up) > 0.9 else ""))
    log("%d normal families" % len(families))
    return families


def loadFamilies(work):
    return np.load(os.path.join(work, FAMILIES_FILE))


def upFamily(families, scene):
    """The family closest to the camera up, oriented up."""
    upCam = scene.meanUp()
    up = families[np.argmax(np.abs(families @ upCam))]
    return up * np.sign(up @ upCam)
