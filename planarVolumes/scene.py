"""Tracked scene: pinhole cameras, poses, 3D track points and their 2D observations, in the working frame.

The working frame is the SfM frame scaled by `scale`, so that the median scene depth equals a reference depth: the
thresholds of the pipeline are expressed in this unit ("u"). Images are referenced by their frame number (the last
digits of their file name), as in a shot.
"""
import os

import numpy as np

# Track points used to anchor monocular depth: seen in at least this many views
MIN_TRACK_LENGTH = 3


def frameNumber(name):
    """Frame number of an image: the last group of digits of its file name (-1 if none)."""
    digits = "".join(ch if ch.isdigit() else " " for ch in os.path.splitext(os.path.basename(name))[0]).split()
    return int(digits[-1]) if digits else -1


class Scene:
    """Cameras and track of a shot in the working frame.

    Attributes:
        names: image file name per view (sorted by frame).
        frames: frame number per view.
        K: (n, 4) pinhole intrinsics (fx, fy, cx, cy) per view, in pixels.
        sizes: (n, 2) image size (width, height) per view.
        R, t: (n, 3, 3) and (n, 3) world-to-camera rotation and translation (x_cam = R x + t).
        points, colors, trackLengths: 3D track points, their RGB colors and number of observations.
        observations: per view, (xy (k, 2) pixel positions, point indices (k,)).
        imagesFolder: folder of the undistorted images.
        scale: working frame = SfM frame * scale.
    """

    def __init__(self, names, K, sizes, R, t, points, colors, trackLengths, observations, imagesFolder="", scale=1.0):
        order = np.argsort([frameNumber(n) for n in names], kind="stable")
        self.names = [names[i] for i in order]
        self.frames = [frameNumber(n) for n in self.names]
        self.K = np.asarray(K, float)[order]
        self.sizes = np.asarray(sizes, int)[order]
        self.R = np.asarray(R, float)[order]
        self.t = np.asarray(t, float)[order]
        self.points = np.asarray(points, float).reshape(-1, 3)
        self.colors = np.asarray(colors, np.uint8).reshape(-1, 3)
        self.trackLengths = np.asarray(trackLengths, int)
        self.observations = [observations[i] for i in order]
        self.imagesFolder = imagesFolder
        self.scale = float(scale)
        self._index = {f: i for i, f in enumerate(self.frames)}
        self._imageFiles = None

    # --- views
    def index(self, frame):
        return self._index[frame]

    def hasFrame(self, frame):
        return frame in self._index

    def center(self, frame):
        i = self._index[frame]
        return -self.R[i].T @ self.t[i]

    def centers(self, frames=None):
        frames = self.frames if frames is None else frames
        return np.array([self.center(f) for f in frames])

    def project(self, frame, X):
        """Pixel positions (n, 2) and depths (n,) of world points X (n, 3) in a view."""
        i = self._index[frame]
        Xc = X @ self.R[i].T + self.t[i]
        fx, fy, cx, cy = self.K[i]
        z = Xc[:, 2]
        return np.stack([fx * Xc[:, 0] / z + cx, fy * Xc[:, 1] / z + cy], 1), z

    def meanUp(self, step=10):
        """Mean 'up' direction of the cameras (-y of the image), unit vector."""
        up = -np.mean([self.R[self._index[f]][1] for f in self.frames[::step]], 0)
        return up / np.linalg.norm(up)

    def trackObservations(self, frame):
        """Reliable track points seen in a view: pixel positions (k, 2) and world positions (k, 3)."""
        xy, idx = self.observations[self._index[frame]]
        good = self.trackLengths[idx] >= MIN_TRACK_LENGTH
        return xy[good], self.points[idx[good]]

    def reliablePoints(self):
        return self.points[self.trackLengths >= MIN_TRACK_LENGTH]

    # --- images
    def imagePath(self, frame):
        """Image file of a view in imagesFolder, matched by file stem (the extension may differ from the SfM)."""
        if self._imageFiles is None:
            self._imageFiles = {}
            if self.imagesFolder and os.path.isdir(self.imagesFolder):
                for f in os.listdir(self.imagesFolder):
                    self._imageFiles.setdefault(os.path.splitext(f)[0], f)
        stem = os.path.splitext(self.names[self._index[frame]])[0]
        f = self._imageFiles.get(stem, self.names[self._index[frame]])
        return os.path.join(self.imagesFolder, f)

    def image(self, frame):
        import cv2
        img = cv2.imread(self.imagePath(frame), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.imagePath(frame))
        return img

    # --- persistence
    def save(self, path):
        xy = [o[0] for o in self.observations]
        ids = [o[1] for o in self.observations]
        counts = np.array([len(i) for i in ids], np.int64)
        np.savez_compressed(path, names=np.array(self.names), K=self.K, sizes=self.sizes, R=self.R, t=self.t,
                            points=self.points, colors=self.colors, trackLengths=self.trackLengths,
                            obsCounts=counts, obsXy=np.concatenate(xy) if xy else np.zeros((0, 2)),
                            obsIds=np.concatenate(ids) if ids else np.zeros(0, np.int64),
                            imagesFolder=np.array(self.imagesFolder), scale=np.array(self.scale))

    @classmethod
    def load(cls, path):
        with np.load(path) as npz:
            z = {k: npz[k] for k in npz.files}          # each access to an npz entry decompresses it: read once
        ends = np.cumsum(z["obsCounts"])
        starts = ends - z["obsCounts"]
        xy, ids = z["obsXy"], z["obsIds"]
        obs = [(xy[a:b], ids[a:b]) for a, b in zip(starts, ends)]
        return cls(list(z["names"]), z["K"], z["sizes"], z["R"], z["t"], z["points"], z["colors"], z["trackLengths"],
                   obs, str(z["imagesFolder"]), float(z["scale"]))


def medianSceneDepth(R, t, points, observations):
    """Median over the views of the median depth of the points they observe (SfM frame)."""
    med = []
    for Ri, ti, (_, idx) in zip(R, t, observations):
        if len(idx) >= 30:
            z = (points[idx] @ Ri.T + ti)[:, 2]
            z = z[z > 0]
            if len(z) >= 30:
                med.append(np.median(z))
    return float(np.median(med)) if med else 0.0


def fromSfmData(sfmPath, imagesFolder, referenceDepth=0.0, scale=None, log=print):
    """Load an AliceVision SfMData (undistorted pinhole cameras, e.g. the output of IntrinsicsTransforming used with the
    images of ExportImages) into a Scene in the working frame.

    Args:
        sfmPath: SfMData file (.sfm, .abc).
        imagesFolder: folder of the undistorted images (matched by file stem).
        referenceDepth: the world is scaled so that the median scene depth equals this value (0: no scaling).
        scale: imposed scale (has priority over referenceDepth).

    Returns:
        Scene.
    """
    from pyalicevision import sfmData as avSfmData
    from pyalicevision import sfmDataIO as avSfmDataIO
    data = avSfmData.SfMData()
    if not avSfmDataIO.load(data, sfmPath, avSfmDataIO.ALL):
        raise RuntimeError("Cannot load SfMData: %s" % sfmPath)
    intrinsics = {}
    for iid, intr in data.getIntrinsics().items():
        p = list(intr.getParameters())
        if intr.getTypeStr() != "pinhole":
            log("intrinsic %d of type %s: used as a pinhole camera" % (iid, intr.getTypeStr()))
        if hasattr(intr, "hasDistortion") and intr.hasDistortion():
            log("intrinsic %d has distortion: ignored (use undistorted images)" % iid)
        w, h = intr.w(), intr.h()
        intrinsics[iid] = ((p[0], p[1], 0.5 * w + p[2], 0.5 * h + p[3]), (w, h))
    viewIds, names, K, sizes, R, C = [], [], [], [], [], []
    for vid, v in data.getViews().items():
        if not data.isPoseAndIntrinsicDefined(v):
            continue
        T = data.getPose(v).getTransform()
        viewIds.append(vid)
        names.append(os.path.basename(v.getImage().getImagePath()))
        k, wh = intrinsics[v.getIntrinsicId()]
        K.append(k)
        sizes.append(wh)
        R.append(np.array(T.rotation()).reshape(3, 3))
        C.append(np.array(T.center()).reshape(3))
    R, C = np.array(R), np.array(C)
    viewIndex = {vid: i for i, vid in enumerate(viewIds)}
    points, colors, tracks = [], [], []
    for _, landmark in data.getLandmarks().items():
        points.append(np.array(landmark.X).reshape(3))
        colors.append(np.array(landmark.rgb).reshape(3))
        tracks.append([viewIndex[v] for v in landmark.getMapObservations().keys() if v in viewIndex])
    points = np.array(points).reshape(-1, 3)
    byView = [[] for _ in viewIds]
    for i, tr in enumerate(tracks):
        for j in tr:
            byView[j].append(i)
    t = -np.einsum("nij,nj->ni", R, C)
    # 2D observations: pinhole projection of each point in the views that observe it (undistorted images)
    observations = []
    for j in range(len(viewIds)):
        idx = np.array(byView[j], np.int64)
        if len(idx):
            Xc = points[idx] @ R[j].T + t[j]
            ok = Xc[:, 2] > 1e-9
            idx, Xc = idx[ok], Xc[ok]
            fx, fy, cx, cy = K[j]
            xy = np.stack([fx * Xc[:, 0] / Xc[:, 2] + cx, fy * Xc[:, 1] / Xc[:, 2] + cy], 1)
        else:
            xy = np.zeros((0, 2))
        observations.append((xy, idx))
    log("%d intrinsics, %d posed views, %d points" % (len(intrinsics), len(viewIds), len(points)))
    if scale:
        s = float(scale)
        log("imposed scale %.6g" % s)
    else:
        d = medianSceneDepth(R, t, points, observations)
        s = referenceDepth / d if (referenceDepth > 0 and d > 0) else 1.0
        log("median scene depth %.4g -> scale %.6g" % (d, s))
    return Scene(names, K, sizes, R, s * t, s * points, colors, [len(tr) for tr in tracks], observations,
                 imagesFolder, s)
