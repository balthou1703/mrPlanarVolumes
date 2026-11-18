__version__ = "1.0"
__author__ = "Balthazar Sahel"
__license__ = "MPL-2.0"

import os

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

STEPS = ["depth", "families", "segments", "planes", "carving", "refinement", "finalization", "stairs", "largePlanes",
         "recesses", "denseStructure", "output"]


class PlanarVolumes(desc.Node):
    """Clean, low-poly layout of a tracked or photogrammetry scene: one slab per wall and floor, openings cut as simple
    shapes (rectangles, arches, circles) with their reveals, stairs as steps.

    Steps: monocular geometry of key frames (MoGe-2, field of view of the tracked camera) aligned on the track ->
    normal families -> planar segments per frame anchored on the track points -> planes per domain -> extent by
    multi-view free-space voting -> photometric refinement (plane sweep) -> large planes in the dominant frame
    (ghost walls in foliage removed, walls extended along the dense mesh) -> recess map of each wall -> openings as
    primitives (crossed by the camera = through) -> dense mesh structure (missing planes, stairs as steps).

    Domains split the shot into parts seen from one side of the walls (e.g. outside then inside): they are processed
    separately, then merged. Indoors, window panes are openings and walls go down to the floor.

    Connect:
    - SfMData: the tracking, with undistorted pinhole cameras (IntrinsicsTransforming);
    - Images Folder: the matching undistorted images (ExportImages);
    - Dense Mesh (optional): the photogrammetry mesh (MeshFiltering): ghost walls, wall extents, missing planes, steps;
    - Person / Window Masks (optional): ImageSegmentationPrompt (mrSegmentation) with the prompts "person" and "window".
    """

    # Credits of the wrapped method (shown in the node info, with the plugin author and license)
    __nodeInfo__ = [
        ("method", "MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details "
                   "(R. Wang, S. Xu, Y. Dong et al., arXiv:2507.02546, 2025)"),
        ("methodLicense", "MIT (code), Apache-2.0 (DINOv2 backbone)"),
        ("methodRepository", "https://github.com/microsoft/MoGe"),
    ]

    category = "Geometry"
    gpu = desc.Level.INTENSIVE
    ram = desc.Level.INTENSIVE
    cpu = desc.Level.INTENSIVE

    inputs = [
        desc.File(name="input", label="SfMData", description="Tracking with undistorted pinhole cameras.", value=""),
        desc.File(name="imagesFolder", label="Images Folder",
                  description="Undistorted images of the SfMData (matched by file name stem).", value=""),
        desc.File(name="denseMesh", label="Dense Mesh", description="Dense mesh of the scene (Meshroom frame), optional.",
                  value=""),
        desc.File(name="personMasks", label="Person Masks",
                  description="Folder of person masks, one per image named after it, non-zero on people. Optional.",
                  value=""),
        desc.File(name="windowMasks", label="Window Masks",
                  description="Folder of window masks, one per image named after it, non-zero on window panes. Optional.",
                  value=""),
        desc.StringParam(name="domains", label="Domains",
                         description="Parts of the shot seen from one side of the walls: 'name:first-last[:interior],...' "
                                     "with frame numbers, e.g. 'ext:950-2460,int:2200-3450:interior'. "
                                     "Empty: one outdoor domain over the whole shot.",
                         value=""),
        desc.BoolParam(name="stairs", label="Stairs",
                       description="Look for stairs in the first outdoor domain (straight stairs from the monocular "
                                   "normals, refined into steps with the dense mesh).",
                       value=False),
        desc.IntParam(name="depthStep", label="Key Frame Step",
                      description="Monocular geometry (MoGe-2) on 1 frame out of N.", value=10, range=(1, 100, 1)),
        desc.FloatParam(name="referenceDepth", label="Reference Depth",
                        description="The scene is scaled so that its median depth equals this value: the thresholds of "
                                    "the steps are tuned for it (working unit). Outputs are back in the SfM frame.",
                        value=2.95, range=(0.1, 100.0, 0.05), advanced=True),
        desc.File(name="mogeModel", label="MoGe-2 Model", description="MoGe-2 checkpoint (normal variant).",
                  value="${PLANARVOLUMES_MOGE_MODEL}", advanced=True),
        desc.ChoiceParam(name="firstStep", label="First Step",
                         description="Resume from this step (the previous ones must be in the working folder).",
                         value=STEPS[0], values=STEPS, exclusive=True, advanced=True, invalidate=False),
        desc.ChoiceParam(name="lastStep", label="Last Step", description="Stop after this step.",
                         value=STEPS[-1], values=STEPS, exclusive=True, advanced=True, invalidate=False),
        desc.ChoiceParam(name="verboseLevel", label="Verbose Level", description="Verbosity level.",
                         value="info", values=VERBOSE_LEVEL, exclusive=True),
    ]

    outputs = [
        desc.File(name="output", label="Folder", description="Working folder.", value="{nodeCacheFolder}"),
        desc.File(name="volumes", label="Volumes", semantic="3d",
                  description="Holed walls with reveals and glazing, floors, stairs, dense mesh planes (Meshroom frame).",
                  value="{nodeCacheFolder}/volumes.obj"),
        desc.File(name="planes", label="Detailed Planes", semantic="3d",
                  description="All the refined plane rectangles (Meshroom frame).", value="{nodeCacheFolder}/planes.obj"),
    ]

    def processChunk(self, chunk):
        import subprocess
        import sys
        chunk.logManager.start(chunk.node.verboseLevel.value)
        node = chunk.node
        log = chunk.logger.info
        try:
            if not node.input.value or not node.imagesFolder.value:
                raise RuntimeError("SfMData and Images Folder are required.")
            work = node.output.value
            os.makedirs(work, exist_ok=True)
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sys.path.insert(0, root)
            from planarVolumes import scene as pvScene
            scenePath = os.path.join(work, "scene.npz")
            scene = pvScene.fromSfmData(node.input.value, node.imagesFolder.value, node.referenceDepth.value, log=log)
            scene.save(scenePath)
            # The steps run in a separate process: torch and the AliceVision libraries loaded by Meshroom may clash.
            cmd = [sys.executable, "-m", "planarVolumes.pipeline", "--work", work, "--scene", scenePath,
                   "--domains", node.domains.value, "--depth-step", str(node.depthStep.value),
                   "--moge-model", node.mogeModel.evalValue, "--dense-mesh", node.denseMesh.value,
                   "--person-masks", node.personMasks.value, "--window-masks", node.windowMasks.value,
                   "--from", node.firstStep.value, "--to", node.lastStep.value]
            if node.stairs.value:
                cmd.append("--stairs")
            env = dict(os.environ)
            # The steps do not use AliceVision: its libraries are kept out of the search path of torch / OpenCV
            env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if "alicevision" not in p.lower())
            env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")])
            env["PYTHONIOENCODING"] = "utf-8"
            log("command: " + " ".join('"%s"' % c if " " in c else c for c in cmd))
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=root,
                                 encoding="utf-8", errors="replace")
            for line in p.stdout:
                log(line.rstrip())
            if p.wait():
                raise RuntimeError("PlanarVolumes failed (code %d)" % p.returncode)
        finally:
            chunk.logManager.end()
