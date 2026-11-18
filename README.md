# mrPlanarVolumes

Meshroom plugin that turns a tracked shot (or a photogrammetry) into a **clean, low-poly layout** of the scene: one
slab per wall and floor in the dominant frame of the building, openings cut as simple shapes (rectangles, arches,
circles) with their reveals and glazing, stairs as steps. The goal is a stable base for layout and set extension,
not a remodelling: few vertices, but the surfaces the camera passes close to keep their real shape (a door the
camera goes through is a through opening with its arch).

| Node | Step | Output |
|------|------|--------|
| `PlanarVolumes` | SfM + undistorted images (+ dense mesh, + person / window masks) | `volumes.obj` (holed walls, reveals, glazing, floors, stairs), `planes.obj` (all the refined plane rectangles), in the Meshroom frame |

---

## Method

1. **Monocular geometry** of key frames (1 out of N) with [MoGe-2](https://github.com/microsoft/MoGe), the field of
   view imposed by the tracked camera; depth aligned per frame on the track points (robust affine fit).
2. **Normal families**: dominant orientations of the surfaces (greedy mode search on the sphere).
3. **Planar segments** per frame, anchored on the track points they contain.
4. **Planes per domain** (parts of the shot seen from one side of the walls, e.g. outside then inside).
5. **Extent by free-space voting**: each cell of a plane is voted surface / seen-through / occluded in every key frame
   (vegetation and people do not vote, window panes are openings indoors).
6. **Photometric refinement**: plane sweep with multi-view NCC (distance and tilts); planes that are not
   photo-consistent (foliage, false planes) are dropped.
7. **Large planes**: one slab per surface in the dominant frame (facade normal, side, up), refined over its whole
   extent. Ghost walls are removed (an unverified outdoor wall mostly seen through, or without dense surface on it),
   walls are extended along the dense mesh up to the corners.
8. **Floors** under the camera path: height map of the lowest points (track + dense mesh) around the path; the
   levels are the heights right under the cameras (furniture is walked around, not on), one slab per level cut at the
   walls; levels guessed from the stairs (terrace, ground) take the measured height.
9. **Recess map** of each wall: depth of the real surface behind the plane, over all the key frames.
10. **Openings**: primitives (rectangle, arch, circle) by sequential extraction with model selection, regularized
   (common sills, springings and widths); an opening crossed by the camera path goes through the wall.
11. **Dense structure** (optional dense mesh): missing planes (turrets, retaining walls, balustrades) and stairs as
    steps (treads at regular heights, any plan shape).

All thresholds are expressed in a working unit: the scene is scaled so that its median depth equals the
*Reference Depth*; the outputs are back in the SfM frame.

---

## Requirements

- **[Meshroom](https://github.com/alicevision/Meshroom)** develop (2026.1 or later), with the plugin manager.
- **NVIDIA GPU** with a driver supporting CUDA 12.6 for the monocular depth step; the other steps run on the CPU.
- **Python** 3.10 or newer (the Python of Meshroom).

Python dependencies are declared in [`pyproject.toml`](pyproject.toml): NumPy 1.x (pyalicevision ABI), OpenCV,
SciPy, PyTorch and [MoGe](https://github.com/microsoft/MoGe) (MoGe-2, pinned commit). The MoGe-2 ViT-L normal weights
are downloaded as a plugin asset (`weights/moge-2-vitl-normal.pt`, 1.3 GB).

## Installation

From a plugin registry: `meshroom_plugins install mrPlanarVolumes`. From a clone: add the folder to
`MESHROOM_PLUGINS_PATH` and create its environment with `uv sync` (`UV_PROJECT_ENVIRONMENT=venv`), then download the
weights listed in `pyproject.toml`.

## Usage

Typical graph, after a camera tracking:

```
CameraInit -> ... -> StructureFromMotion -> IntrinsicsTransforming (pinhole) -> PlanarVolumes.input
                                         -> ExportImages (undistorted)        -> PlanarVolumes.imagesFolder
              ... -> Meshing -> MeshFiltering                                 -> PlanarVolumes.denseMesh (optional)
ImageSegmentationPrompt ("person")  (mrSegmentation)                         -> PlanarVolumes.personMasks (optional)
ImageSegmentationPrompt ("window")  (mrSegmentation)                         -> PlanarVolumes.windowMasks (optional)
```

Main parameters:

- **Domains**: `name:first-last[:interior],...` with frame numbers, e.g. `ext:950-2460,int:2200-3450:interior` for a
  shot going from outside to inside. Empty: one outdoor domain.
- **Stairs**: look for stairs in the first outdoor domain.
- **Key Frame Step**: monocular depth on 1 frame out of N (10 by default). The monocular step is the longest one
  (about 1 s per key frame on a recent GPU); the plane sweep takes a few minutes per domain.

## Tests

CPU only, no network, no weights:

```
venv/Scripts/python -m pytest tests        (Windows)
venv/bin/python -m pytest tests            (Linux)
```

## License

[MPL-2.0](LICENSE). MoGe is under the MIT license (DINOv2 backbone: Apache-2.0).
