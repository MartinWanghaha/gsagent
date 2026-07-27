"""End-to-end Semantic Prior Field pipeline.

Runs joint geometry + semantic training with the Semantic Prior Field
enabled, then pivot-based mesh extraction with semantic edge filtering and
vertex semantics, then mesh texturing.

Usage:
    python semantic_prior_field/scripts/train_and_extract_spf.py \
        -s <COLMAP_DATASET> -m <OUTPUT_DIR> \
        --semantic_masks <ASSOCIATED_MASK_DIR> \
        [--rasterizer ours|radegs] [-r 2] [--no_postprocess]

Any unrecognized flag is forwarded to train.py (step 1).
"""

import argparse
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train.py")
EXTRACT_SCRIPT = os.path.join(BASE_DIR, "pivot_based_mesh_extraction.py")
TEXTURE_SCRIPT = os.path.join(BASE_DIR, "texture_mesh.py")

MESH_NAME = "mesh_ours_2pivots.ply"

TRAIN_FLAGS = [
    "--feature_dc_lr", "0.0013",
    "--feature_rest_lr", "0.00011",
    "--exposure_compensation",
    "--data_device", "cpu",
    "--N_max_gaussians", "6000000",
    "--semantic_prior",
]

# --filter_semantic_edges is deliberately NOT enabled: in cluttered scenes
# with fine-grained instances it fragments the mesh (see docs/experiments.md
# AB-MESH). Vertex identity export (--use_semantics) is kept: zero risk.
EXTRACT_FLAGS = [
    "--sdf_mode", "ours",
    "--dtype", "int32",
    "--isosurface_value", "0.0",
    "--n_binary_steps", "10",
    "--iteration", "30000",
    "--use_valid_mask",
    "--postprocess",
    "--filter_large_edges",
    "--use_semantics",
    "--data_device", "cpu",
]


def parse_data_args(args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-s", "--source_path")
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-r", "--resolution")
    known, _ = parser.parse_known_args(args)
    result = []
    if known.source_path:
        result += ["-s", known.source_path]
    if known.model_path:
        result += ["-m", known.model_path]
    if known.resolution:
        result += ["-r", known.resolution]
    return result, known


parser_ext = argparse.ArgumentParser(add_help=False)
parser_ext.add_argument("--rasterizer", choices=["ours", "radegs"], default="ours")
parser_ext.add_argument("--no_postprocess", action="store_true", help="Disable the postprocessing step.")
parser_ext.add_argument("--skip_texturing", action="store_true", help="Skip the mesh texturing step.")
ext_args, user_args = parser_ext.parse_known_args(sys.argv[1:])

if "--semantic_masks" not in user_args:
    print("[ERROR] --semantic_masks <dir> is required: the Semantic Prior Field "
          "is derived from the associated Gaga masks.")
    sys.exit(1)

TRAIN_FLAGS = ["--rasterizer", ext_args.rasterizer] + TRAIN_FLAGS
EXTRACT_FLAGS = ["--rasterizer", ext_args.rasterizer] + EXTRACT_FLAGS
if ext_args.no_postprocess and "--postprocess" in EXTRACT_FLAGS:
    EXTRACT_FLAGS.remove("--postprocess")

shared_args, known = parse_data_args(user_args)

print("[INFO] Step 1/3: Training with Semantic Prior Field...")
result = subprocess.run([sys.executable, TRAIN_SCRIPT] + TRAIN_FLAGS + user_args)
if result.returncode != 0:
    sys.exit(result.returncode)

print("[INFO] Step 2/3: Extracting mesh with semantic edge filtering...")
extract_args = [arg for arg in shared_args if not arg.startswith("--semantic_masks")]
result = subprocess.run([sys.executable, EXTRACT_SCRIPT] + EXTRACT_FLAGS + extract_args)
if result.returncode != 0:
    sys.exit(result.returncode)

if ext_args.skip_texturing:
    print("[INFO] Skipping texturing step.")
    sys.exit(0)

print("[INFO] Step 3/3: Refining texture...")
mesh_path = os.path.join(known.model_path, MESH_NAME) if known.model_path else MESH_NAME
if "--postprocess" in EXTRACT_FLAGS:
    mesh_path = mesh_path.replace(".ply", "_post.ply")
result = subprocess.run(
    [sys.executable, TEXTURE_SCRIPT, "--rasterizer", ext_args.rasterizer, "--mesh", mesh_path]
    + shared_args
)
sys.exit(result.returncode)
