"""Monocular geometry per key frame (MoGe-2: depth, camera-space normals, validity mask), aligned on the track.

Key frames are stored in <work>/depth/<frame %04d>.npz (float16 depth and normals, boolean mask, horizontal FOV).
"""
import glob
import os
import time

import cv2
import numpy as np

from .geometry import robustDepthScale

DEPTH_DIR = "depth"


def depthPath(work, frame):
    return os.path.join(work, DEPTH_DIR, "%04d.npz" % frame)


def depthFrames(work):
    """Frames with a monocular depth file, sorted."""
    return sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(work, DEPTH_DIR, "*.npz")))


def loadDepth(work, frame):
    """Monocular depth (H, W) float32, camera-space normals (H, W, 3) float32 and validity mask (H, W) of a frame."""
    z = np.load(depthPath(work, frame))
    return z["depth"].astype(np.float32), z["normal"].astype(np.float32), z["mask"]


def runMoge(scene, work, frames, modelPath, log=print):
    """MoGe-2 inference on key frames, with the field of view imposed by the tracked camera.

    Args:
        scene: Scene.
        work: working folder.
        frames: frames to process (existing results are kept).
        modelPath: MoGe-2 checkpoint (model.pt) or Hugging Face model id.
    """
    import torch
    from moge.model.v2 import MoGeModel
    os.makedirs(os.path.join(work, DEPTH_DIR), exist_ok=True)
    todo = [f for f in frames if not os.path.exists(depthPath(work, f))]
    log("monocular depth: %d key frames (%d to compute)" % (len(frames), len(todo)))
    if not todo:
        return
    device = torch.device("cuda")
    model = MoGeModel.from_pretrained(modelPath).to(device).eval().half()
    t0 = time.time()
    for i, frame in enumerate(todo):
        img = cv2.cvtColor(scene.image(frame), cv2.COLOR_BGR2RGB)
        W = img.shape[1]
        fx = scene.K[scene.index(frame)][0]
        fovX = np.degrees(2 * np.arctan(W / (2 * fx)))
        x = torch.tensor(img / 255.0, dtype=torch.float16, device=device).permute(2, 0, 1)
        with torch.no_grad():
            r = model.infer(x, fov_x=fovX)
        np.savez_compressed(depthPath(work, frame), depth=r["depth"].float().cpu().numpy().astype(np.float16),
                            normal=r["normal"].float().cpu().numpy().astype(np.float16),
                            mask=r["mask"].cpu().numpy(), fov_x=fovX)
        if i % 20 == 0:
            log("  frame %d (%d/%d), %.0f s" % (frame, i + 1, len(todo), time.time() - t0))
    log("monocular depth: %d frames in %.0f s" % (len(todo), time.time() - t0))


def trackScale(scene, frame, depth, valid, mask=None):
    """Affine scale (a, b) mapping a monocular depth map to the working frame, from the track points of the view.

    Args:
        mask: optional (H, W) boolean: only the track points inside are used.

    Returns:
        tuple: ((a, b), track depths of the used points, number of used points).
    """
    R, t = scene.R[scene.index(frame)], scene.t[scene.index(frame)]
    xy, Xw = scene.trackObservations(frame)
    H, W = depth.shape
    zc = (Xw @ R.T + t)[:, 2]
    u = np.clip(xy[:, 0].astype(int), 0, W - 1)
    v = np.clip(xy[:, 1].astype(int), 0, H - 1)
    zm = depth[v, u]
    ok = np.isfinite(zm) & valid[v, u] & (zc > 0)
    if mask is not None:
        ok &= mask[v, u]
    return robustDepthScale(zm[ok], zc[ok]), zc[ok], int(ok.sum())


def alignedGeometry(scene, work, frame, ds=2, mask=None):
    """Monocular geometry of a key frame in the working frame, at 1/ds resolution.

    Returns:
        dict: R, t, K (fx, fy, cx, cy at 1/ds), dd (depth map, inf where invalid), N (world normals),
        X (world points), C (camera center), ab (affine depth scale), nalign (track points used).
    """
    depth, normal, valid = loadDepth(work, frame)
    i = scene.index(frame)
    R, t = scene.R[i], scene.t[i]
    H, W = depth.shape
    (a, b), _, nAlign = trackScale(scene, frame, depth, valid, mask)
    h, w = H // ds, W // ds
    dd = cv2.resize(np.nan_to_num(depth, nan=np.inf, posinf=np.inf), (w, h), interpolation=cv2.INTER_NEAREST) * a + b
    mk = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    dd[~mk] = np.inf
    nr = cv2.resize(normal, (w, h), interpolation=cv2.INTER_AREA)
    nr /= np.linalg.norm(nr, axis=-1, keepdims=True) + 1e-9
    fx, fy, cx, cy = scene.K[i]
    K = (fx / ds, fy / ds, (cx + 0.5) / ds - 0.5, (cy + 0.5) / ds - 0.5)
    vv, uu = np.mgrid[0:h, 0:w]
    Pc = np.stack([(uu - K[2]) / K[0] * dd, (vv - K[3]) / K[1] * dd, dd], -1)
    return dict(R=R, t=t, K=K, dd=dd, N=nr @ R, X=(Pc - t) @ R, C=-R.T @ t, ab=(a, b), nalign=nAlign)
