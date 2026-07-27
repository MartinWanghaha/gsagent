# Tanks and Temples Evaluation Toolbox: Uniform and Virtual Scan Sampling evaluations.

Python scripts for evaluating reconstructions against the [Tanks and Temples](https://tanksandtemples.org/) benchmark. This repository contains the evaluations presented in the ["From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians"](https://diego1401.github.io/BlobsToSpokesWebsite/index.html) work. For the legacy evaluation please refer to this [link](https://github.com/Anttwo/MILo/tree/master/milo/eval/tnt).

## Installation

```bash
conda create -n uniscan_tnt_eval python=3.9 -y
conda activate uniscan_tnt_eval
pip install -r requirements.txt
# nvdiffrast is required for virtual_scan_sampling_eval.py only (needs a CUDA-enabled GPU)
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
```

## Dataset Setup

Download the evaluation data (ground truth geometry + reference reconstruction) and place it so that each scene folder contains:

```
<dataset_dir>/
  <Scene>.ply           # Ground truth point cloud
  <Scene>.json          # Crop volume
  <Scene>_trans.txt     # Alignment transform
  <Scene>_COLMAP_SfM.log
```

---

## Evaluation Scripts

### Uniform Sampling Evaluation (`uniform_sampling_eval.py`)

Evaluates a mesh or point cloud against the ground truth using surface-based sampling and a 3-step ICP alignment pipeline.

```bash
python uniform_sampling_eval.py \
    --dataset-dir <path/to/scene_dir> \
    --traj-path <path/to/scene_dir>/<Scene>_COLMAP_SfM.log \
    --ply-path <path/to/mesh.ply> \
    --out-dir <output_dir>
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--save_point_clouds` | off | Save color-coded precision/recall `.ply` files |

---

### Virtual Scan Sampling Evaluation (`virtual_scan_sampling_eval.py`)

Evaluates a mesh by rendering depth from each training camera and projecting to world-space point clouds, simulating a real scan. Requires a COLMAP dataset (Blender scenes are not supported).

```bash
python virtual_scan_sampling_eval.py \
    -s <path/to/colmap_dataset> \
    -r 2 \
    --dataset-dir <path/to/scene_dir> \
    --traj-path <path/to/scene_dir>/<Scene>_COLMAP_SfM.log \
    --ply-path <path/to/mesh.ply> \
    --out-dir <output_dir>
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--save_point_clouds` | off | Save color-coded precision/recall `.ply` files |

---

## Output

Results are written to `--out-dir` and include:

- Console output with **precision**, **recall**, and **F-score**
- Precision/recall curve plots (PDF)
- Optionally (`--save_point_clouds`): `<Scene>.precision.ply` / `<Scene>.recall.ply` (color-coded error clouds)