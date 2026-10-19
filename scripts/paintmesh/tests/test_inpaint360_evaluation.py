"""Test benchmark pairing, the author's masking rule, and scene aggregation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "eval_inpaint360.py"
spec = importlib.util.spec_from_file_location("inpaint360_evaluation", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def save_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8)).save(path)


def test_masking_uses_white_region_and_keeps_full_image_denominator():
    mask = np.array([[0, 128], [127, 255]], dtype=np.uint8)
    image = np.full((2, 2, 3), 100, dtype=np.uint8)
    masked = evaluation.masked_array(image, mask)
    assert np.array_equal(masked[0, 0], [0, 0, 0])
    assert np.array_equal(masked[1, 0], [0, 0, 0])
    assert np.array_equal(masked[0, 1], [100, 100, 100])
    assert np.mean(masked.astype(float) ** 2) == 5000
    # It is not an MSE normalized by only the selected pixels.
    assert np.mean(masked.astype(float) ** 2) != 10000


def test_image_tensor_supports_author_psnr_flattening():
    array = np.full((8, 12, 3), 100, dtype=np.uint8)
    tensor = evaluation.image_tensor(array, "cpu")
    assert tensor.shape == (1, 3, 8, 12)
    assert tensor.is_contiguous()
    assert float((tensor ** 2).view(1, -1).mean()) == pytest.approx((100 / 255) ** 2)


def test_pairs_reference_stems_across_different_extensions(tmp_path):
    for folder in ["renders", "gt", "masks"]:
        for name in ["test_B", "test_A"]:
            save_image(tmp_path / folder / (name + (".JPG" if folder == "gt" else ".png")))
    save_image(tmp_path / "masks/IMG_training.png")
    pairs = evaluation.match_images(tmp_path / "renders", tmp_path / "gt", tmp_path / "masks")
    assert [pair[0] for pair in pairs] == ["test_A", "test_B"]


def test_missing_masks_are_errors_instead_of_skipped_views(tmp_path):
    for folder in ["renders", "gt", "masks"]:
        for name in ["test_A", "test_B"]:
            save_image(tmp_path / folder / (name + ".png"))
    (tmp_path / "masks/test_B.png").unlink()
    with pytest.raises(ValueError, match="match exactly"):
        evaluation.match_images(tmp_path / "renders", tmp_path / "gt", tmp_path / "masks")


def test_duplicate_stems_are_rejected(tmp_path):
    save_image(tmp_path / "test_A.png")
    save_image(tmp_path / "test_A.JPG")
    with pytest.raises(ValueError, match="duplicate image stem"):
        evaluation.image_index(tmp_path)


def result(path, scene, frames, value, fid, resolution=4):
    metrics = {name: value for name in evaluation.METRICS}
    metrics["FID"] = fid
    evaluation.write_json(path, {
        "complete": True, "inputs": {"scene": scene}, "frame_count": frames,
        "protocol": dict(evaluation.PROTOCOL, evaluation_resolution=resolution),
        "metrics": metrics,
    })
    return path


def test_aggregation_distinguishes_image_mean_from_scene_mean(tmp_path):
    paths = [result(tmp_path / "a.json", "a", 2, 10, 100),
             result(tmp_path / "b.json", "b", 6, 30, 200)]
    combined = evaluation.aggregate_results(paths)
    assert combined["official_code_average"]["PSNR"] == 25
    assert combined["scene_macro_average"]["PSNR"] == 20
    assert combined["official_code_average"]["FID"] == 150
    assert combined["frame_count"] == 8


def test_aggregation_rejects_different_resolutions(tmp_path):
    paths = [result(tmp_path / "a.json", "a", 2, 10, 100),
             result(tmp_path / "b.json", "b", 2, 20, 200, resolution=2)]
    with pytest.raises(ValueError, match="different metric definitions"):
        evaluation.aggregate_results(paths)


def test_aggregation_rejects_multiple_variants_of_one_scene(tmp_path):
    paths = [result(tmp_path / "a.json", "bag", 2, 10, 100),
             result(tmp_path / "b.json", "bag", 2, 20, 200)]
    with pytest.raises(ValueError, match="one result per scene"):
        evaluation.aggregate_results(paths)


def test_aggregation_rejects_mixed_training_policies(tmp_path):
    paths = [result(tmp_path / "a.json", "a", 2, 10, 100),
             result(tmp_path / "b.json", "b", 2, 20, 200)]
    data = json.loads(paths[1].read_text())
    data["inputs"]["training_eval"] = True
    evaluation.write_json(paths[1], data)
    with pytest.raises(ValueError, match="different training policies"):
        evaluation.aggregate_results(paths)
