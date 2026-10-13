"""Exercise scene resolution and batch dispatch without reconstruction or CUDA."""

import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "run_seg.sh"
OUTPUT_OVERRIDES = (
    "PIPELINE_ROOT", "EDGS_MODEL_ROOT", "EDGS_BRIDGE_ROOT",
    "SEMANTIC_GS_ROOT", "SEMANTIC_MESH_ROOT",
)


@pytest.fixture
def batch(tmp_path):
    entry = tmp_path / "scripts/paintmesh/batch_entry.sh"
    entry.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, entry)
    child = entry.with_name("run_seg.sh")
    child.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\t%s\\t%s\\t%s\\n" "$@" >> "$BATCH_CALL_LOG"\n'
        'if [[ "$2" == "${FAIL_SCENE:-}" ]]; then exit 23; fi\n'
    )
    child.chmod(0o755)
    for folder in ["submodules/EDGS", "submodules/Inpaint360GS", "ckpt"]:
        (tmp_path / folder).mkdir(parents=True)
    config = tmp_path / "submodules/Inpaint360GS/config/object_distill/train_distill.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"iterations": 2000}')
    (tmp_path / "ckpt/CropFormer_hornet_3x_03823a.pth").touch()
    dataset = tmp_path / "data/testset"
    dataset.mkdir(parents=True)
    log = tmp_path / "calls.tsv"
    env = os.environ.copy()
    for variable in OUTPUT_OVERRIDES + (
        "DATA_ROOT", "CKPT_ROOT", "OUTPUT_ROOT", "EDGS_IMAGES", "FAIL_SCENE",
        "EDGS_SAVE_ITERATIONS", "DISTILL_SAVE_ITERATIONS",
        "DATASET_TYPE", "EVAL", "DISTILL_DATA_DEVICE",
    ):
        env.pop(variable, None)
    env.update(CONDA_BIN="/bin/true", BATCH_CALL_LOG=str(log))

    def scene(name):
        root = dataset / name
        for directory in ["images", "images_2", "sparse/0"]:
            (root / directory).mkdir(parents=True)
        for name in ["cameras.bin", "images.bin", "points3D.bin"]:
            (root / "sparse/0" / name).touch()
        return root

    def run(dataset_name="testset", scene_name="all", **overrides):
        return subprocess.run(
            ["bash", str(entry), dataset_name, scene_name, "2", "1"],
            cwd=tmp_path, env={**env, **overrides},
            text=True, capture_output=True,
        )

    def calls():
        return [line.split("\t") for line in log.read_text().splitlines()] if log.exists() else []

    return dataset, scene, run, calls


@pytest.fixture
def training_probe(tmp_path):
    """Stop at the training command after recording its actual arguments."""
    executable = tmp_path / "conda-probe"
    arguments = tmp_path / "training-arguments.txt"
    executable.write_text(
        '#!/usr/bin/env bash\n'
        'for argument in "$@"; do\n'
        '  if [[ "$argument" == train.py ]]; then\n'
        '    printf "%s\\n" "$@" > "$TRAINING_ARGUMENTS"\n'
        '    exit 42\n'
        '  fi\n'
        'done\n'
        'exit 0\n'
    )
    executable.chmod(0o755)
    return {"CONDA_BIN": str(executable), "TRAINING_ARGUMENTS": str(arguments)}, arguments


@pytest.mark.parametrize("relative_scene", [
    "inpaint360/bag", "group with spaces/bag with spaces", "inpaint360/bag[1]",
])
def test_scene_shorthand_matches_explicit_training_and_output_paths(batch, training_probe, relative_scene):
    dataset, scene, run, _ = batch
    scene(relative_scene)
    dataset = dataset.rename(dataset.with_name("Inpaint360GS"))
    probe_env, arguments = training_probe
    shorthand = run("Inpaint360GS", Path(relative_scene).name, **probe_env)
    assert shorthand.returncode == 42, shorthand.stderr
    short_arguments = arguments.read_text().splitlines()
    explicit = run("Inpaint360GS", relative_scene, **probe_env)
    assert explicit.returncode == 42, explicit.stderr
    assert short_arguments == arguments.read_text().splitlines()
    output = dataset.parents[1] / "output/paintmesh/Inpaint360GS" / relative_scene
    assert f"gs.dataset.source_path={dataset / relative_scene}" in short_arguments
    assert f"gs.dataset.model_path={output / 'edgs'}" in short_arguments
    assert "gs.dataset.resolution=2" in short_arguments
    assert "gs.dataset.eval=false" in short_arguments
    assert "gs.dataset.exclude_test_images=true" in short_arguments
    assert shorthand.stdout == explicit.stdout


def test_direct_scene_takes_precedence_over_grouped_matches(batch, training_probe):
    dataset, scene, run, _ = batch
    for name in ["bag", "inpaint360/bag", "others/bag"]:
        scene(name)
    probe_env, arguments = training_probe
    result = run(scene_name="bag", **probe_env)
    assert result.returncode == 42, result.stderr
    assert f"gs.dataset.source_path={dataset / 'bag'}" in arguments.read_text().splitlines()


def test_ambiguous_shorthand_fails_before_outputs_or_training(batch, training_probe):
    dataset, scene, run, _ = batch
    for name in ["inpaint360/bag", "others/bag"]:
        scene(name)
    probe_env, arguments = training_probe
    result = run(scene_name="bag", **probe_env)
    assert result.returncode == 1
    assert "ambiguous scene 'bag'" in result.stderr
    assert "inpaint360/bag" in result.stderr
    assert "others/bag" in result.stderr
    assert not arguments.exists()
    assert not (dataset.parents[1] / "output").exists()
    explicit = run(scene_name="others/bag", **probe_env)
    assert explicit.returncode == 42, explicit.stderr
    assert f"gs.dataset.source_path={dataset / 'others/bag'}" in arguments.read_text().splitlines()


@pytest.mark.parametrize("scene_name", ["missing", "missing/bag"])
def test_missing_scene_does_not_fall_back_to_another_scene(batch, training_probe, scene_name):
    dataset, scene, run, _ = batch
    scene("inpaint360/bag")
    probe_env, arguments = training_probe
    result = run(scene_name=scene_name, **probe_env)
    assert result.returncode == 1
    assert f"required directory does not exist: {dataset / scene_name}" in result.stderr
    assert not arguments.exists()


def test_dispatches_all_scenes_with_resolution_and_stage(batch):
    dataset, scene, run, calls = batch
    scene("kitchen")
    scene("bonsai room")
    (dataset / "notes.txt").touch()
    (dataset / "metadata").mkdir()
    result = run(OUTPUT_ROOT="custom-output")
    assert result.returncode == 0, result.stderr
    assert calls() == [
        ["testset", "bonsai room", "2", "1"],
        ["testset", "kitchen", "2", "1"],
    ]


@pytest.mark.parametrize("missing", ["images_2", "sparse/0/images.bin"])
def test_checks_every_scene_before_starting_any_training(batch, missing):
    _, scene, run, calls = batch
    scene("a-ready")
    root = scene("z-incomplete")
    path = root / missing
    path.rmdir() if path.is_dir() else path.unlink()
    result = run()
    assert result.returncode != 0
    assert "z-incomplete" in result.stderr
    assert calls() == []


def test_preserves_failure_status_and_stops_later_scenes(batch):
    _, scene, run, calls = batch
    for name in ["a", "b", "c"]:
        scene(name)
    result = run(FAIL_SCENE="b")
    assert result.returncode == 23
    assert [call[1] for call in calls()] == ["a", "b"]
    assert "batch stopped at scene 'b'" in result.stderr


def test_inpaint360gs_grouped_scenes_exclude_reference_views_and_default_to_no_holdout(batch):
    dataset, scene, run, calls = batch
    scene("inpaint360/doppelherz")
    bear = scene("others/bear")
    (bear / "images_2").rmdir()  # The single-scene runner generates missing scales.
    dataset.rename(dataset.with_name("Inpaint360GS"))
    result = run(dataset_name="Inpaint360GS")
    assert result.returncode == 0, result.stderr
    assert "eval=false; exclude_test_images=true" in result.stdout
    assert calls() == [
        ["Inpaint360GS", "inpaint360/doppelherz", "2", "1"],
        ["Inpaint360GS", "others/bear", "2", "1"],
    ]


def test_eval_override_and_explicit_dataset_type(batch):
    _, scene, run, calls = batch
    scene("doppelherz")
    result = run(DATASET_TYPE="inpaint360", EVAL="true")
    assert result.returncode == 0, result.stderr
    assert "eval=true; exclude_test_images=true" in result.stdout
    assert len(calls()) == 1


def test_colmap_retains_default_holdout(batch):
    _, scene, run, _ = batch
    scene("kitchen")
    result = run()
    assert result.returncode == 0, result.stderr
    assert "eval=true; exclude_test_images=false" in result.stdout


@pytest.mark.parametrize("variable", OUTPUT_OVERRIDES)
def test_rejects_output_paths_shared_across_scenes(batch, variable):
    _, scene, run, calls = batch
    scene("a")
    result = run(**{variable: "shared-model"})
    assert result.returncode != 0
    assert variable in result.stderr
    assert calls() == []


def test_no_scenes_returns_error(batch):
    _, _, run, calls = batch
    result = run()
    assert result.returncode != 0
    assert "no scene directories" in result.stderr
    assert calls() == []


def test_reserved_all_directory_cannot_recurse(batch):
    _, scene, run, calls = batch
    scene("all")
    result = run()
    assert result.returncode != 0
    assert "reserved" in result.stderr
    assert calls() == []


def test_save_iterations_are_sorted_deduplicated_and_include_final_steps(batch):
    _, scene, run, calls = batch
    scene("kitchen")
    result = run(
        EDGS_SAVE_ITERATIONS="9000,3000,9000",
        DISTILL_SAVE_ITERATIONS="1500,500",
    )
    assert result.returncode == 0, result.stderr
    assert "EDGS PLY save iterations: 3000 9000 30000" in result.stdout
    assert "Semantic PLY save iterations: 500 1500 2000" in result.stdout
    assert len(calls()) == 1


@pytest.mark.parametrize("variable,value", [
    ("EDGS_SAVE_ITERATIONS", "30001"),
    ("DISTILL_SAVE_ITERATIONS", "2001"),
    ("EDGS_SAVE_ITERATIONS", "9999999999999999999999999999999999999999"),
    ("EDGS_SAVE_ITERATIONS", "0"),
    ("DISTILL_SAVE_ITERATIONS", "500, 1000"),
    ("EDGS_SAVE_ITERATIONS", "3000,,6000"),
    ("DISTILL_SAVE_ITERATIONS", "500,$(touch injected)"),
])
def test_invalid_save_iterations_never_start_training(batch, variable, value):
    _, scene, run, calls = batch
    scene("kitchen")
    result = run(**{variable: value})
    assert result.returncode != 0
    assert variable in result.stderr
    assert calls() == []


def test_running_pipeline_keeps_parsed_code_when_script_is_rewritten(tmp_path):
    """Simulate editing the runner while a long subprocess has paused it."""
    entry = tmp_path / "running.sh"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    prelude = (
        'printf ready > "$TEST_READY"\n'
        'while [[ ! -e "$TEST_RELEASE" ]]; do sleep 0.01; done\n'
        'printf "Snapshot completed\\n"\n'
        'return 0\n'
    )
    source = SCRIPT.read_text()
    assert "main() {\n" in source
    entry.write_text(source.replace("main() {\n", "main() {\n" + prelude, 1))
    env = {**os.environ, "TEST_READY": str(ready), "TEST_RELEASE": str(release)}
    process = subprocess.Popen(
        ["bash", str(entry)], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "runner did not reach the simulated long subprocess"
        # This replacement is deliberately invalid Bash and larger than the
        # original file, so rereading even its trailing bytes would fail.
        entry.write_text(")\n" * (len(source) + 10000))
        release.touch()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr
        assert "Snapshot completed" in stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
