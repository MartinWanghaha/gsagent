#!/usr/bin/env python3
"""Validate a reusable EDGS correspondence initializer without importing models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _plain(value: Any) -> Any:
    return (
        OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
    )


_LEGACY_LIST_FIELDS = {
    "coarse_resolution",
    "prematch_resolution",
    "target_resolution",
}


def _normalize_manifest_config(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Normalize schema-v2 manifests written before ListConfig serialization was fixed."""

    if isinstance(value, dict):
        return {
            key: _normalize_manifest_config(item, (*path, key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_manifest_config(item, path) for item in value]
    if (
        isinstance(value, str)
        and path
        and path[-1] in _LEGACY_LIST_FIELDS
        and value.startswith("[")
        and value.endswith("]")
    ):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(decoded, list):
            return _normalize_manifest_config(decoded, path)
    return value


def _nested(value: Any, path: str) -> Any:
    current = value
    for name in path.split("."):
        if not isinstance(current, dict) or name not in current:
            raise KeyError(path)
        current = current[name]
    return current


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_group_budget(identity: dict[str, Any]) -> int:
    camera_count = len(identity.get("camera_names", ()))
    planner = identity.get("config", {}).get("group_planner", {})
    policy = str(planner.get("budget_policy", "one_per_source"))
    if policy in {"auto", "one_per_source"}:
        return camera_count
    if policy == "paper_half":
        return max(camera_count, math.ceil(0.5 * camera_count * math.sqrt(camera_count)))
    if policy == "paper_full":
        return max(camera_count, math.ceil(camera_count * math.sqrt(camera_count)))
    if policy == "fixed":
        return int(planner["fixed_budget"])
    raise SystemExit(f"unsupported group budget policy in manifest: {policy!r}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=("roma", "mvroma"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="DOTTED_PATH=JSON",
        help="Compare a resolved config value without duplicating defaults.",
    )
    parser.add_argument(
        "--expect-string",
        action="append",
        default=[],
        metavar="DOTTED_PATH=VALUE",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = _plain(OmegaConf.load(args.config))
    if not isinstance(config, dict) or not isinstance(config.get("init_wC"), dict):
        raise SystemExit(f"saved EDGS config has no init_wC mapping: {args.config}")
    init_config = config["init_wC"]
    if init_config.get("use") is not True:
        raise SystemExit("saved EDGS run did not use correspondence initialization")
    backend = str(init_config.get("backend", "roma")).lower()
    if backend != args.backend:
        raise SystemExit(
            f"correspondence backend mismatch: saved={backend}, requested={args.backend}"
        )

    mismatches = []
    expectations = []
    for expression in args.expect:
        if "=" not in expression:
            raise SystemExit(f"invalid --expect expression: {expression!r}")
        path, encoded = expression.split("=", 1)
        expectations.append((path, json.loads(encoded)))
    for expression in args.expect_string:
        if "=" not in expression:
            raise SystemExit(f"invalid --expect-string expression: {expression!r}")
        path, expected = expression.split("=", 1)
        expectations.append((path, expected))
    for path, expected in expectations:
        try:
            actual = _nested(config, path)
        except KeyError:
            mismatches.append(f"{path}: missing, requested={expected!r}")
            continue
        if actual != expected or type(actual) is not type(expected):
            mismatches.append(f"{path}: saved={actual!r}, requested={expected!r}")
    if mismatches:
        raise SystemExit(
            "saved correspondence run belongs to different inputs or parameters:\n  "
            + "\n  ".join(mismatches)
        )

    if backend == "roma":
        print("EDGS correspondence contract: roma")
        return

    manifest_path = args.manifest or args.config.parent / "correspondence_init" / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            "MV-RoMa schema-v2 manifest is missing; use a new output directory: "
            f"{manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("kind") != "edgs-mvroma-correspondence-init"
        or manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
    ):
        raise SystemExit(f"invalid or incomplete MV-RoMa manifest: {manifest_path}")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise SystemExit(f"MV-RoMa manifest has no identity: {manifest_path}")
    if _normalize_manifest_config(identity.get("config")) != init_config.get("mvroma"):
        raise SystemExit(
            "MV-RoMa manifest/config mismatch; the output is not safe to reuse"
        )
    expected_artifact = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    if manifest.get("artifact_id") != expected_artifact:
        raise SystemExit("MV-RoMa manifest artifact_id is invalid")
    if identity.get("effective_primary_group_budget") != _expected_group_budget(identity):
        raise SystemExit("MV-RoMa manifest records an inconsistent group budget")
    files = manifest.get("files", {})
    hashes = manifest.get("file_sha256", {})
    if set(files) != {"groups", "overlap", "pair_quality"} or set(hashes) != set(files):
        raise SystemExit("MV-RoMa manifest has an incomplete sidecar contract")
    for name, relative in files.items():
        sidecar = manifest_path.parent / str(relative)
        if not sidecar.is_file():
            raise SystemExit(f"MV-RoMa sidecar is missing: {relative}")
        if _sha256_file(sidecar) != hashes[name]:
            raise SystemExit(f"MV-RoMa sidecar SHA-256 mismatch: {relative}")

    groups = json.loads((manifest_path.parent / files["groups"]).read_text(encoding="utf-8"))
    plan = [
        [int(group["source_index"]), list(group["target_indices"])]
        for group in groups.get("groups", ())
    ]
    plan_digest = hashlib.sha256(_canonical_bytes(plan)).hexdigest()
    if plan_digest != identity.get("group_plan_digest"):
        raise SystemExit("MV-RoMa groups sidecar does not match its plan digest")
    print(
        "EDGS correspondence contract: mvroma/"
        f"{identity.get('algorithm_version')} ({expected_artifact[:12]})"
    )


if __name__ == "__main__":
    main()
