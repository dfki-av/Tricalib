# TriCalib

Interactive extrinsic calibration tool for RGB camera, LiDAR, and event camera.

## Overview

TriCalib provides a GUI-based workflow to compute the spatial transformations between three sensor modalities. Two calibration approaches are supported:

**Pairwise calibration** — solve each sensor pair independently:
- **RGB camera** ↔ **LiDAR** — via PnP (Perspective-n-Point)
- **Event camera** ↔ **LiDAR** — via PnP
- **RGB camera** ↔ **Event camera** — via essential matrix decomposition

**Joint optimization** — optimize all three modality pairs simultaneously via non-linear least-squares over reprojection errors across all pairs.

**Key features:**
- Interactive 2D/3D point correspondence selection (right-click in image and point cloud viewers)
- Real-time projection visualization with depth/intensity coloring
- Reprojection error reporting per modality pair
- Session save/load as JSON (full calibration state)

## Requirements

- **Python 3.10** (recommended; 3.9 and 3.12+ are not supported)
- Git with Git LFS

## Installation

This repository uses [Git LFS](https://git-lfs.com/) for large binary assets (point clouds, images in `examples/`).

```bash
# Install Git LFS binary (once per machine)
# macOS:   brew install git-lfs
# Linux:   sudo apt install git-lfs  (or equivalent)
# Windows: download installer from https://git-lfs.com
git lfs install

# Clone the repository
git clone git@pc-4501.kl.dfki.de:jakkamsetty/tricalib.git
cd tricalib

# Pull LFS objects
git lfs pull

# Install in editable mode
pip install -e .
```

## Usage

Launch the GUI:

```bash
python -m tricalib
```

**Workflow:**
1. Load sensor data — RGB image, event image, LiDAR `.pcd` file, and camera intrinsics JSON
2. Select corresponding 3D/2D point pairs across modalities using the point cloud and image viewers
3. Run calibration — either pairwise (PnP or essential matrix per sensor pair) or joint optimization (all pairs simultaneously)
4. Inspect projections in the viewer windows and check reprojection errors
6. Save the calibration session to JSON for later use

## Input Data

| File | Format | Description |
|------|--------|-------------|
| RGB image | `.png` / `.jpg` | Frame from the RGB camera |
| Event image | `.png` | Accumulated event frame |
| Point cloud | `.pcd` | LiDAR scan (Open3D compatible) |
| Intrinsics | `.json` | Camera intrinsic matrices and distortion coefficients for RGB and event cameras |

See `examples/data/dfki/dfki_intrinsics.json` for the expected intrinsics format.

## Examples

Two example datasets are included under `examples/data/`:

- `dfki/` — DFKI dataset with pre-labeled point correspondences and a saved session
- `dsec/` — Two sequences from the DSEC dataset (`000044/`, `000058/`)

To try the DFKI example, load the files from `examples/data/dfki/` and import `points_000236.json` as correspondences, or load the full saved state from `session.json`.

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Authors

Rahul Jakkamsetty — [DFKI](https://www.dfki.de) (German Research Center for Artificial Intelligence)
