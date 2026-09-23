import argparse
import json
import os
import re
import stat
import tempfile
from pathlib import Path


CONFIG_NAMES = ("object_inpaint", "object_removal")


def _unique(values):
    """Return integer IDs in input order, without duplicates."""
    return list(dict.fromkeys(values or []))


def _validate_scene_key(dataset_name, scene):
    dataset_text = str(dataset_name)
    dataset_path = Path(dataset_text)
    if (
        not dataset_text
        or dataset_path.is_absolute()
        or any(part in {"", ".", ".."} for part in dataset_text.split("/"))
    ):
        raise ValueError("dataset_name must be a safe relative path")
    scene_text = str(scene)
    if not scene_text or scene_text in {".", ".."} or Path(scene_text).name != scene_text:
        raise ValueError("scene must be one safe path component")


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_mode = (
        stat.S_IMODE(path.stat().st_mode)
        if path.exists()
        else None
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if previous_mode is not None:
            os.chmod(temporary, previous_mode)
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def setup_configs(
    dataset_name,
    scene,
    target_id,
    surrounding_ids,
    output_root=None,
    removal_thresh=None,
):
    """
    Create scene configs from the repository templates.

    ``output_root`` may be used to isolate generated configs from the source
    tree. It should be the directory that will contain ``object_inpaint`` and
    ``object_removal``. When omitted, the historical ``config`` destination is
    retained.
    """

    target_ids = _unique(target_id)
    _validate_scene_key(dataset_name, scene)
    if not target_ids:
        print("[!] Error: target_id is required.")
        return []
    if removal_thresh is not None and not 0.0 <= removal_thresh <= 1.0:
        raise ValueError("removal_thresh must be between 0 and 1")

    # A surrounding object that is already a removal target must not be added
    # twice. Keep the user-provided order so generated configs are stable.
    target_id_set = set(target_ids)
    safe_surrounding = [
        obj_id for obj_id in _unique(surrounding_ids)
        if obj_id not in target_id_set
    ]
    final_obj_ids = target_ids + safe_surrounding

    repository_config_root = Path(__file__).resolve().parents[1] / "config"
    destination_root = (
        Path(output_root).expanduser()
        if output_root is not None
        else repository_config_root
    )
    destination_root = destination_root.resolve()

    generated_paths = []
    for config_name in CONFIG_NAMES:
        folder = destination_root / config_name / dataset_name
        try:
            folder.resolve(strict=False).relative_to(destination_root)
        except ValueError as exc:
            raise ValueError(f"generated config directory escapes output_root: {folder}") from exc
        folder.mkdir(parents=True, exist_ok=True)
        # Templates are always read from the repository, including when output
        # is redirected to an isolated run directory.
        source_file = repository_config_root / config_name / "common.json"
        target_file = folder / f"{scene}.json"

        initialized = not target_file.exists()
        if not source_file.exists():
            print(f"[!] Warning: {source_file} not found. Skipping {folder}.")
            continue

        try:
            input_file = source_file if initialized else target_file
            try:
                with input_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if initialized or output_root is None:
                    raise
                print(
                    f"[!] Recovering malformed run-local config {target_file} "
                    f"from {source_file}: {exc}"
                )
                with source_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

            data["target_id"] = target_ids
            data["surrounding_ids"] = safe_surrounding
            data["select_obj_id"] = final_obj_ids
            if removal_thresh is not None:
                data["removal_thresh"] = removal_thresh

            full_json = json.dumps(data, indent=4, ensure_ascii=False)
            compact_json = re.sub(
                r"\[\s+([\d,\s]+)\s+\]",
                lambda match: "[" + ", ".join(match.group(1).split()).replace(",,", ",") + "]",
                full_json,
            )

            _atomic_write_text(target_file, compact_json)

            generated_paths.append(target_file)
            if initialized:
                print(f"[+] Initialized {target_file} from template.")
            print(f"[*] Successfully updated {target_file}: select_obj_id -> {final_obj_ids}")

        except Exception as e:
            print(f"[!] Error processing {target_file}: {e}")

    return generated_paths


def list_of_ints(arg):
    if not arg or str(arg).lower() == "none":
        return []
    return [int(x.strip()) for x in str(arg).replace(',', ' ').split()]


def unit_interval(arg):
    value = float(arg)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return value


def main():
    parser = argparse.ArgumentParser(description="Configure scene JSONs with clean multi-line formatting.")
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset_name")
    parser.add_argument("--scene", type=str, required=True, help="Scene name")
    parser.add_argument("--target_id", type=list_of_ints, required=True, help="Primary object ID")
    parser.add_argument(
        "--target_surronding_id",
        "--target_surrounding_id",
        dest="target_surrounding_id",
        type=list_of_ints,
        default=None,
        help="Surrounding object IDs (e.g., 1,2,3). The misspelled legacy option is also supported.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional destination config root; templates are still read from the repository config directory.",
    )
    parser.add_argument(
        "--removal_thresh",
        type=unit_interval,
        default=None,
        help="Override the removal probability threshold in both generated configs (0 to 1).",
    )

    args = parser.parse_args()
    try:
        generated = setup_configs(
            args.dataset_name,
            args.scene,
            args.target_id,
            args.target_surrounding_id,
            output_root=args.output_root,
            removal_thresh=args.removal_thresh,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if len(generated) != len(CONFIG_NAMES):
        raise SystemExit("failed to generate both object-removal and object-inpaint configs")


if __name__ == "__main__":
    main()
