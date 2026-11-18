"""Semantic masks per image: people (never planes, no votes) and window panes (seen-through openings indoors).

The masks come from segmentation nodes (e.g. ImageSegmentationPrompt of the mrSegmentation plugin with the prompts
"person" and "window"): one image per view, named after the image file stem, non-zero on the object.
"""
import os

import cv2
import numpy as np

# Window detections larger than this fraction of the image are false positives (whole facade)
MAX_WINDOW_FRACTION = 0.2


class SemanticMasks:
    """Person and window masks of the views, read on demand from two folders (either may be empty).

    Args:
        scene: the Scene (image names).
        personFolder: folder of person masks.
        windowFolder: folder of window masks.
    """

    def __init__(self, scene, personFolder="", windowFolder=""):
        self.scene = scene
        self.folders = {"person": personFolder, "window": windowFolder}
        self._files = {}
        self._cache = {}

    def _file(self, kind, frame):
        folder = self.folders[kind]
        if not folder or not os.path.isdir(folder):
            return None
        if kind not in self._files:
            self._files[kind] = {}
            for f in os.listdir(folder):
                stem, ext = os.path.splitext(f)
                if ext.lower() in (".png", ".exr", ".jpg", ".tif", ".tiff") and not stem.startswith("bboxes_"):
                    self._files[kind].setdefault(stem, os.path.join(folder, f))
        stem = os.path.splitext(self.scene.names[self.scene.index(frame)])[0]
        return self._files[kind].get(stem)

    def available(self):
        return any(self.folders.values())

    def _read(self, kind, frame):
        key = (kind, frame)
        if key not in self._cache:
            path = self._file(kind, frame)
            m = None
            if path:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
                if img is not None:
                    if img.ndim == 3:
                        img = img.max(-1)
                    m = img.astype(np.float32) > (0.5 if img.dtype.kind == "f" else 127)
                    if kind == "window":
                        m = windowHulls(m)
            self._cache[key] = m
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def labels(self, frame, shape=None):
        """Label image of a view: 0 nothing, 1 person, 2 window (people painted over windows), or None.

        Args:
            frame: frame number.
            shape: optional (height, width) to resize to (nearest).
        """
        person, window = self._read("person", frame), self._read("window", frame)
        if person is None and window is None:
            return None
        ref = person if person is not None else window
        out = np.zeros(ref.shape, np.uint8)
        if window is not None:
            out[cv2.resize(window.astype(np.uint8), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST) > 0] = 2
        if person is not None:
            out[person] = 1
        if shape is not None and out.shape != tuple(shape):
            out = cv2.resize(out, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return out


def windowHulls(mask):
    """Window panes: convex hull of each detected window (mullions and reflections included), huge detections dropped."""
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros(mask.shape, bool)
    H, W = mask.shape
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > MAX_WINDOW_FRACTION * W * H:
            continue
        pts = cv2.findNonZero((lab == i).astype(np.uint8))
        hull = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
        out |= hull > 0
    return out


class LabelFolder:
    """Masks already merged in one label image per frame (<folder>/<frame %04d>.png: 0, 1 person, 2 window)."""

    def __init__(self, *folders):
        self.folders = [f for f in folders if f]

    def available(self):
        return any(os.path.isdir(f) for f in self.folders)

    def labels(self, frame, shape=None):
        for folder in self.folders:
            path = os.path.join(folder, "%04d.png" % frame)
            if os.path.exists(path):
                out = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if shape is not None and out.shape != tuple(shape):
                    out = cv2.resize(out, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                return out
        return None


class NoMasks:
    def available(self):
        return False

    def labels(self, frame, shape=None):
        return None
