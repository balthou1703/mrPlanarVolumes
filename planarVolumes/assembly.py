"""Final outputs: detailed planes (all the finalized rectangles) and volumes (holed large planes + dense structure),
in the working frame and in the Meshroom frame (x, -y, -z, SfM scale).
"""
import os
import pickle

import numpy as np

from .geometry import MESHROOM_FLIP, quadsToObjects, writeObj


def detailedPlanes(work, domains, stairs, log=print):
    """All the finalized rectangles of the domains (+ straight stairs). The rectangles of an outdoor domain seen only
    from inside are removed (duplicates of the interior domain).

    Returns:
        list of quads (V, name).
    """
    firstInterior = min([fmin for _, fmin, _, interior in domains if interior], default=None)
    quads = []
    for name, _, _, interior in domains:
        with open(os.path.join(work, name, "final.pkl"), "rb") as f:
            final = pickle.load(f)
        Q = final["quads"]
        if not interior and firstInterior is not None:
            Q = [q for q in Q if min(q["frames"]) < firstInterior - 200]
        kind = "interior" if interior else "exterior"
        quads += [dict(V=q["V"], name="%s_%03d%s" % (kind, j, "" if q.get("refined", True) else "_unrefined"))
                  for j, q in enumerate(Q)]
        log("%s (%s): %d rectangles of %d" % (name, kind, len(Q), len(final["quads"])))
    if stairs is not None:
        quads += [dict(V=q["V"], name="stairs_%03d" % j) for j, q in enumerate(stairs["quads"])]
    return quads


def mergeVolumes(volumes, dense):
    """Volumes + dense structure; when the dense mesh gave the stairs, they replace the straight stairs."""
    if dense is None:
        return list(volumes)
    if dense["stairs"]:
        volumes = [q for q in volumes if not q["name"].startswith("stairs")]
    return list(volumes) + list(dense["quads"])


def writeOutputs(quads, workPath, meshroomPath, scale, header):
    """Write quads to the working frame OBJ and to the Meshroom frame OBJ (x, -y, -z, divided by the scene scale)."""
    objects = quadsToObjects(quads)
    writeObj(workPath, objects, header=header + " - working frame")
    return writeObj(meshroomPath, objects, header=header + " - Meshroom frame", flip=MESHROOM_FLIP, scale=1.0 / scale)


def quadCount(quads):
    return len(quads), int(np.sum([len(q["V"]) for q in quads])) if quads else 0
