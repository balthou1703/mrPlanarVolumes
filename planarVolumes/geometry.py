"""Small geometric helpers shared by the pipeline steps: rectangles on occupancy grids, periodic combs, OBJ files."""
import numpy as np

# Frame conversions. The working frame is the SfM frame (AliceVision convention, camera looking along +z, y down)
# scaled by the scene scale. Meshroom meshes and its 3D viewer use (x, -y, -z).
MESHROOM_FLIP = np.array([1.0, -1.0, -1.0])


def maxRect(mask):
    """Largest axis-aligned rectangle of ones in a binary grid (histogram of heights per row).

    Args:
        mask: 2D array, non-zero cells are occupied.

    Returns:
        tuple: (area, x0, y0, x1, y1) in cells, x1 and y1 excluded; area is 0 when the grid is empty.
    """
    heights = np.zeros(mask.shape[1], int)
    best = (0, 0, 0, 0, 0)
    for y in range(mask.shape[0]):
        heights = np.where(mask[y], heights + 1, 0)
        stack = []
        for x in range(len(heights) + 1):
            cur = heights[x] if x < len(heights) else 0
            start = x
            while stack and stack[-1][1] >= cur:
                s0, hh = stack.pop()
                area = hh * (x - s0)
                if area > best[0]:
                    best = (area, s0, y - hh + 1, x, y + 1)
                start = s0
            stack.append((start, cur))
    return best


def greedyRects(mask, maxRects=6, stopFraction=0.15):
    """Greedy decomposition of a binary grid into maximal rectangles.

    Args:
        mask: 2D boolean grid (not modified).
        maxRects: maximum number of rectangles.
        stopFraction: stop when the uncovered part is below this fraction of the initial occupancy.

    Returns:
        list: (row0, row1, col0, col1) per rectangle, ends excluded.
    """
    m = np.asarray(mask, bool).copy()
    total = m.sum()
    out = []
    for _ in range(maxRects):
        if m.sum() < stopFraction * total or m.sum() == 0:
            break
        area, x0, y0, x1, y1 = maxRect(m)
        if area == 0:
            break
        out.append((y0, y1, x0, x1))
        m[y0:y1, x0:x1] = False
    return out


def rectSubtract(rect, cut):
    """Boolean difference of two axis-aligned rectangles (b0, b1, a0, a1).

    Returns:
        list: rectangles covering rect minus cut (at most 4), slivers thinner than 0.05 dropped.
    """
    b0, b1, a0, a1 = rect
    c0, c1, d0, d1 = cut
    if c0 >= b1 or c1 <= b0 or d0 >= a1 or d1 <= a0:
        return [rect]
    out = []
    if c0 > b0:
        out.append((b0, c0, a0, a1))
    if c1 < b1:
        out.append((c1, b1, a0, a1))
    m0, m1 = max(b0, c0), min(b1, c1)
    if d0 > a0:
        out.append((m0, m1, a0, d0))
    if d1 < a1:
        out.append((m0, m1, d1, a1))
    return [o for o in out if o[1] - o[0] > 0.05 and o[3] - o[2] > 0.05]


def rectGap(r1, r2):
    """Distance between two rectangles (b0, b1, a0, a1): 0 when they touch or overlap."""
    return max(r1[0] - r2[1], r2[0] - r1[1], r1[2] - r2[3], r2[2] - r1[3], 0)


def rectsOverlap(r1, r2):
    return min(r1[1], r2[1]) - max(r1[0], r2[0]) > 0 and min(r1[3], r2[3]) - max(r1[2], r2[2]) > 0


def comb(values, weights, minStep, maxStep, nbSteps=400):
    """Best period of a 1D comb: maximizes the weighted phase concentration (Rayleigh statistic).

    Args:
        values: positions (e.g. heights of horizontal points).
        weights: weight per position.
        minStep, maxStep: searched period range.

    Returns:
        tuple: (concentration in [0, 1], period, phase in [0, period)).
    """
    best = (0, None, None)
    for step in np.linspace(minStep, maxStep, nbSteps):
        z = np.sum(weights * np.exp(2j * np.pi * values / step))
        r = np.abs(z) / weights.sum()
        if r > best[0]:
            best = (r, step, (np.angle(z) / (2 * np.pi)) * step % step)
    return best


def weightedMedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, c[-1] / 2)]


def weightedQuantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, q * c[-1])]


def robustDepthScale(monoDepth, trackDepth, iterations=6):
    """Robust affine fit trackDepth ~ a * monoDepth + b (relative residuals, Cauchy weights).

    Monocular depth is defined up to an affine transform; the tracked points give its metric (working frame) scale.

    Returns:
        tuple: (a, b); (1, 0) with fewer than 10 samples.
    """
    x, y = monoDepth, trackDepth
    w = np.ones_like(x)
    a, b = 1.0, 0.0
    if len(x) < 10:
        return a, b
    for _ in range(iterations):
        A = np.stack([x, np.ones_like(x)], 1) * np.sqrt(w / y ** 2)[:, None]
        a, b = np.linalg.lstsq(A, y * np.sqrt(w / y ** 2), rcond=None)[0]
        r = np.log(np.maximum(a * x + b, 1e-6) / y)
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        w = 1 / (1 + (r / (2 * s)) ** 2)
    return a, b


def planeLabel(axis, offset):
    """Stable short name of an axis-aligned plane, e.g. ('F', -3.96) -> 'Fm4p0'."""
    return axis + ('%+.1f' % offset).replace('.', 'p').replace('+', '').replace('-', 'm')


def writeObj(path, objects, header=None, flip=None, scale=1.0):
    """Write named polygon groups to an OBJ file.

    Args:
        path: output file.
        objects: iterable of (name, polygons), each polygon an (n, 3) array of vertices (n >= 3, any winding).
        header: optional comment line.
        flip: optional per-axis factors applied to the vertices (e.g. MESHROOM_FLIP).
        scale: factor applied to the vertices (after flip).

    Returns:
        int: number of vertices written.
    """
    f = np.ones(3) if flip is None else np.asarray(flip, float)
    count = 0
    with open(path, "w") as fo:
        if header:
            fo.write("# %s\n" % header)
        for name, polygons in objects:
            fo.write("o %s\n" % name)
            for poly in polygons:
                V = np.asarray(poly, float) * f * scale
                fo.write("".join("v %.5f %.5f %.5f\n" % tuple(v) for v in V))
                fo.write("f " + " ".join(str(count + i + 1) for i in range(len(V))) + "\n")
                count += len(V)
    return count


def readObjVertices(path):
    """Vertices of an OBJ file as an (n, 3) array."""
    V = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append(line.split()[1:4])
    return np.array(V, float).reshape(-1, 3)


def readObj(path):
    """Vertices (n, 3) and triangles (m, 3) of an OBJ file (only the first 3 vertices of each face)."""
    V, F = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append(line.split()[1:4])
            elif line.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V, float).reshape(-1, 3), np.array(F, np.int64).reshape(-1, 3)


def quadsToObjects(quads):
    """Group quads (dicts with 'V' and 'name') into consecutive (name, polygons) OBJ objects."""
    objects = []
    for q in quads:
        if objects and objects[-1][0] == q["name"]:
            objects[-1][1].append(q["V"])
        else:
            objects.append((q["name"], [q["V"]]))
    return objects
