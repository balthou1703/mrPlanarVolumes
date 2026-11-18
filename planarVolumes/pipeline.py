"""PlanarVolumes pipeline: steps, options and command line.

Usage (plugin venv, from the plugin root):
    python -m planarVolumes.pipeline --work <folder> --scene <folder>/scene.npz [--dense-mesh mesh.obj]
        [--person-masks <folder>] [--window-masks <folder>] [--domains "ext:953-2464,int:2204-3455:interior"]
        [--stairs] [--depth-step 10] [--moge-model <model.pt>] [--from STEP] [--to STEP]

The scene file is written by the node from the SfMData (scene.fromSfmData + Scene.save).
"""
import argparse
import os
import sys
import time

from . import assembly, carving, clustering, denseStructure, depth, families, finalization, largePlanes, openings
from . import recesses, refinement, segments, stairs
from .masks import LabelFolder, NoMasks, SemanticMasks
from .scene import Scene

STEPS = ["depth", "families", "segments", "planes", "carving", "refinement", "finalization", "stairs", "largePlanes",
         "recesses", "denseStructure", "output"]


def parseDomains(spec, frames):
    """'ext:953-2464,int:2204-3455:interior' -> [(name, frameMin, frameMax, interior)]; empty: one outdoor domain."""
    if not spec:
        return [("ext", frames[0], frames[-1], False)]
    out = []
    for item in spec.split(","):
        f = item.strip().split(":")
        a, b = f[1].split("-")
        out.append((f[0], int(a), int(b), len(f) > 2 and f[2].startswith("int")))
    return out


class Options:
    """Pipeline options (see the command line)."""

    def __init__(self, work, domains="", stairs=False, depthStep=10, mogeModel="", denseMesh="", personMasks="",
                 windowMasks="", labelMasks=()):
        self.work, self.domains, self.stairs, self.depthStep = work, domains, stairs, depthStep
        self.mogeModel = mogeModel or os.environ.get("PLANARVOLUMES_MOGE_MODEL", "") or "Ruicheng/moge-2-vitl-normal"
        self.denseMesh, self.personMasks, self.windowMasks, self.labelMasks = denseMesh, personMasks, windowMasks, labelMasks


def run(scene, options, first=STEPS[0], last=STEPS[-1], log=print):
    """Run the steps [first, last] of the pipeline.

    Returns:
        dict: paths of the outputs written by the "output" step (volumes, planes), else empty.
    """
    work = options.work
    os.makedirs(work, exist_ok=True)
    doms = parseDomains(options.domains, scene.frames)
    for name, _, _, _ in doms:
        os.makedirs(os.path.join(work, name), exist_ok=True)
    if options.labelMasks:
        masks = LabelFolder(*options.labelMasks)
    elif options.personMasks or options.windowMasks:
        masks = SemanticMasks(scene, options.personMasks, options.windowMasks)
    else:
        masks = NoMasks()
    log("domains: %s; stairs: %s; masks: %s" % (doms, options.stairs, "yes" if masks.available() else "no"))
    selected = STEPS[STEPS.index(first):STEPS.index(last) + 1]
    refiner = None
    outputs = {}
    t0 = time.time()
    for step in selected:
        log("== %s" % step)
        t = time.time()
        if step == "depth":
            depth.runMoge(scene, work, scene.frames[::options.depthStep], options.mogeModel, log)
        elif step == "families":
            families.estimateFamilies(scene, work, log)
        elif step == "segments":
            segments.computeSegments(scene, work, families.loadFamilies(work), masks, log)
        elif step == "planes":
            segs, fams = segments.loadSegments(work), families.loadFamilies(work)
            for name, fmin, fmax, _ in doms:
                clustering.clusterPlanes(scene, segs, fams, os.path.join(work, name), fmin, fmax, log)
        elif step == "carving":
            for name, fmin, fmax, interior in doms:
                carving.carveDomain(scene, work, os.path.join(work, name), fmin, fmax, masks, interior, log)
        elif step == "refinement":
            refiner = refiner or refinement.Refiner(scene, work, masks)
            for name, _, _, _ in doms:
                refinement.refineDomain(refiner, os.path.join(work, name), log)
        elif step == "finalization":
            for name, _, _, interior in doms:
                finalization.finalizeDomain(os.path.join(work, name), interior, log)
        elif step == "stairs":
            outdoor = [d for d in doms if not d[3]]
            if options.stairs and outdoor:
                fits = stairs.fitFrames(scene, work, families.loadFamilies(work), outdoor[0][1], outdoor[0][2], log)
                stairs.buildStairs(fits, work, log)
            elif os.path.exists(os.path.join(work, stairs.STAIRS_FILE)):
                os.remove(os.path.join(work, stairs.STAIRS_FILE))
        elif step == "largePlanes":
            refiner = refiner or refinement.Refiner(scene, work, masks)
            dense = largePlanes.loadDensePoints(options.denseMesh, scene.scale, log)
            largePlanes.LargePlanes(scene, work, doms, refiner, dense, log).run()
        elif step == "recesses":
            recesses.computeRecesses(scene, work, masks, largePlanes.loadLargePlanes(work), log)
        elif step == "denseStructure":
            denseStructure.computeDenseStructure(work, largePlanes.loadLargePlanes(work), stairs.loadStairs(work),
                                                 options.denseMesh, scene.scale, log)
        elif step == "output":
            outputs = writeOutputs(scene, work, doms, log)
        log("   %.0f s" % (time.time() - t))
    log("done in %.1f min" % ((time.time() - t0) / 60))
    return outputs


def loadRecesses(work, large):
    import numpy as np
    out = {}
    for pid in recesses.measuredWalls(large):
        path = os.path.join(work, "recess_%s.npz" % pid)
        if os.path.exists(path):
            out[pid] = dict(np.load(path))
    return out


def writeOutputs(scene, work, doms, log=print):
    """Openings of the measured walls, then volumes (holed large planes, stairs, dense structure) and detailed planes,
    in both frames."""
    import pickle
    large = largePlanes.loadLargePlanes(work)
    straight = stairs.loadStairs(work)
    volumes = openings.holedVolumes(scene, work, large, loadRecesses(work, large), straight, log)
    densePath = os.path.join(work, denseStructure.DENSE_STRUCTURE_FILE)
    dense = None
    if os.path.exists(densePath):
        with open(densePath, "rb") as f:
            dense = pickle.load(f)
        if dense["stairs"]:
            log("stairs from the dense mesh: they replace the straight stairs")
    volumes = assembly.mergeVolumes(volumes, dense)
    paths = dict(volumes=os.path.join(work, "volumes.obj"), planes=os.path.join(work, "planes.obj"),
                 volumesWork=os.path.join(work, "volumes_work.obj"), planesWork=os.path.join(work, "planes_work.obj"))
    nv = assembly.writeOutputs(volumes, paths["volumesWork"], paths["volumes"], scene.scale, "PlanarVolumes volumes")
    log("volumes: %d polygons, %d vertices -> %s" % (len(volumes), nv, paths["volumes"]))
    planes = assembly.detailedPlanes(work, doms, straight, log)
    nv = assembly.writeOutputs(planes, paths["planesWork"], paths["planes"], scene.scale, "PlanarVolumes detailed planes")
    log("detailed planes: %d polygons, %d vertices -> %s" % (len(planes), nv, paths["planes"]))
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="PlanarVolumes: clean low-poly layout of a tracked scene.")
    ap.add_argument("--work", required=True, help="working folder")
    ap.add_argument("--scene", required=True, help="scene file (scene.npz)")
    ap.add_argument("--domains", default="", help="'name:first-last[:interior],...' (default: one outdoor domain)")
    ap.add_argument("--stairs", action="store_true", help="look for straight stairs in the first outdoor domain")
    ap.add_argument("--depth-step", type=int, default=10, help="monocular depth on 1 frame out of N")
    ap.add_argument("--moge-model", default="", help="MoGe-2 checkpoint (default: $PLANARVOLUMES_MOGE_MODEL)")
    ap.add_argument("--dense-mesh", default="", help="dense mesh (Meshroom frame), optional")
    ap.add_argument("--person-masks", default="", help="folder of person masks (named after the images), optional")
    ap.add_argument("--window-masks", default="", help="folder of window masks (named after the images), optional")
    ap.add_argument("--label-masks", default="", help=argparse.SUPPRESS)      # merged label images, comma-separated folders
    ap.add_argument("--from", dest="first", default=STEPS[0], choices=STEPS)
    ap.add_argument("--to", dest="last", default=STEPS[-1], choices=STEPS)
    a = ap.parse_args(argv)
    scene = Scene.load(a.scene)
    options = Options(a.work, a.domains, a.stairs, a.depth_step, a.moge_model, a.dense_mesh, a.person_masks,
                      a.window_masks, tuple(f for f in a.label_masks.split(",") if f))
    run(scene, options, a.first, a.last, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    sys.exit(main())
