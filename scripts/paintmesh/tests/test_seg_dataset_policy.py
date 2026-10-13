"""Exercise view selection with real COLMAP readers and the shell validator."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/paintmesh"))
from prepare_seg_images import prepare_images


def make_scene(root):
    images = root / "images"
    sparse = root / "sparse/0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    names = [f"frame_{i:02d}.png" for i in range(9)] + ["contest.png", "test_removed.png", "TEST_other.png"]
    for name in names:
        Image.new("RGB", (18, 10), color=(70, 80, 90)).save(images / name)
    (sparse / "cameras.txt").write_text("1 PINHOLE 18 10 12 12 9 5\n")
    (sparse / "images.txt").write_text("".join(
        f"{i} 1 0 0 0 {i * 0.1} 0 0 1 {name}\n\n" for i, name in enumerate(names, 1)
    ))
    (sparse / "points3D.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n0 0 1 0 0 1 100 100 100\n"
    )
    return names


@pytest.mark.parametrize("backend", ["edgs", "inpaint"])
@pytest.mark.parametrize("inpaint,eval_enabled,train_count,test_count", [
    (True, False, 10, 0), (True, True, 8, 2), (False, True, 10, 2),
])
def test_readers_exclude_only_reference_images_before_holdout(tmp_path, backend, inpaint, eval_enabled, train_count, test_count):
    scene = tmp_path / ("Inpaint360GS/inpaint360/bag" if inpaint else "mip-nerf/360_v2/kitchen")
    make_scene(scene)
    module_root = ROOT / ("submodules/EDGS/submodules/gaussian-splatting" if backend == "edgs" else "submodules/Inpaint360GS")
    script = """
import json, sys
from types import SimpleNamespace
from scene.dataset_readers import readColmapSceneInfo
path, backend, exclude, evaluate = sys.argv[1:]
exclude, evaluate = exclude == '1', evaluate == '1'
if backend == 'edgs':
    scene = readColmapSceneInfo(path, 'images', '', evaluate, False, exclude_test_images=exclude)
else:
    args = SimpleNamespace(source_path=path, exclude_test_images=exclude, init_mode='sparse')
    scene = readColmapSceneInfo(args, 'images', evaluate, None)
print(json.dumps({'train': [c.image_name for c in scene.train_cameras], 'test': [c.image_name for c in scene.test_cameras]}))
"""
    result = subprocess.run([sys.executable, "-c", script, str(scene), backend, str(int(inpaint)), str(int(eval_enabled))],
        env={**os.environ, "PYTHONPATH": str(module_root)}, capture_output=True, text=True, check=True)
    views = json.loads(result.stdout.splitlines()[-1])
    assert len(views["train"]) == train_count
    assert len(views["test"]) == test_count
    assert "contest" in {Path(name).stem for name in views["train"] + views["test"]}
    if inpaint:
        assert not any(name.lower().startswith("test_") for name in views["train"] + views["test"])


def test_missing_downsamples_use_camera_rounding_and_preserve_existing_images(tmp_path):
    make_scene(tmp_path)
    selected, generated = prepare_images(tmp_path, 4, exclude_test_images=True)
    assert (selected, generated) == (10, 10)
    assert not (tmp_path / "images_4/test_removed.png").exists()
    with Image.open(tmp_path / "images_4/frame_00.png") as image:
        assert image.size == (4, 2)  # round(18/4), round(10/4)
    existing = tmp_path / "images_4/frame_00.png"
    before = existing.stat().st_mtime_ns
    assert prepare_images(tmp_path, 4, exclude_test_images=True) == (10, 0)
    assert existing.stat().st_mtime_ns == before
    assert (tmp_path / "images/test_removed.png").is_file()


def test_mask_validation_uses_same_eligible_views_and_still_rejects_missing_training_masks(tmp_path):
    make_scene(tmp_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    for path in (tmp_path / "images").iterdir():
        if not path.name.lower().startswith("test_"):
            Image.fromarray(np.ones((10, 18), dtype=np.uint16)).save(masks / path.name)
    # Extract the actual shell function, including its embedded Python.
    source = (ROOT / "scripts/paintmesh/run_seg.sh").read_text()
    body = source.split("validate_mask_pairs() {", 1)[1].split("\n}\n", 1)[0]
    def validate(exclude):
        script = f'run_inpaint() {{ {shlex.quote(sys.executable)} "$@"; }}\n'
        script += "validate_mask_pairs() {" + body + "\n}\n"
        script += f'EXCLUDE_TEST_IMAGES={exclude}\nvalidate_mask_pairs {shlex.quote(str(tmp_path / "images"))} {shlex.quote(str(masks))} fixture\n'
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert validate("true").returncode == 0
    assert validate("false").returncode != 0
    (masks / "frame_00.png").unlink()
    result = validate("true")
    assert result.returncode != 0
    assert "frame_00" in result.stderr


def test_cropformer_skips_test_images(tmp_path):
    make_scene(tmp_path)
    # Load only the real loop; the model is a stub so no weights or GPU are needed.
    import ast
    source = (ROOT / "submodules/Inpaint360GS/seg/raw_mask_sam.py").read_text()
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "process_images")
    seen = []
    saved = []
    scope = dict(os=os, IMAGE_EXTENSIONS={".png"}, tqdm=lambda names, **_: names,
                 read_image=lambda path, **_: seen.append(Path(path).name),
                 generate_hqsam_mask=lambda *args: None,
                 save_mask=lambda mask, name, out: saved.append(name),
                 vis_mask_images=lambda *args: None)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "raw_mask_sam.py", "exec"), scope)
    scope["process_images"](None, "hqsam", str(tmp_path / "images"), str(tmp_path / "masks"), .5, True)
    assert len(seen) == len(saved) == 10
    assert "contest.png" in saved
    assert not any(name.lower().startswith("test_") for name in saved)


@pytest.mark.parametrize("stored_eval,stored_exclude,valid", [(False, True, True), (True, True, False), (False, False, False)])
def test_resume_rejects_a_model_trained_with_a_different_view_policy(tmp_path, stored_eval, stored_exclude, valid):
    manifest = tmp_path / "bridge_manifest.json"
    manifest.write_text(json.dumps({
        "dataset": {"source_path": str(tmp_path), "images": "images", "resolution": 4,
                    "eval": stored_eval, "exclude_test_images": stored_exclude},
        "edgs": {"iteration": 30000, "renderer_backend": "pgsr"},
    }))
    source = (ROOT / "scripts/paintmesh/run_seg.sh").read_text()
    body = source.split("validate_bridge_contract() {", 1)[1].split("\nPY\n}\n", 1)[0]
    script = f'run_inpaint() {{ {shlex.quote(sys.executable)} "$@"; }}\n'
    script += "validate_bridge_contract() {" + body + "\nPY\n}\n"
    script += f'SCENE_ROOT={shlex.quote(str(tmp_path))}\nEDGS_IMAGES=images\nRESOLUTION=4\nBASE_ITERATION=30000\nEVAL_OVERRIDE=false\nEXCLUDE_TEST_IMAGES=true\n'
    script += f'validate_bridge_contract {shlex.quote(str(manifest))}\n'
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert (result.returncode == 0) is valid, result.stderr
    if not valid:
        assert "does not match this run" in result.stderr
