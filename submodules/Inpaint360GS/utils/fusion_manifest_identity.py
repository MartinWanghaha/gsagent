"""Stable, content-addressed identity for RGB-D fusion manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

FUSION_MANIFEST_KIND = "paintmesh-rgbd-fusion"
FUSION_IDENTITY_VERSION = 2


def stable_identity_value(value: Any) -> Any:
    """Return JSON-compatible data without volatile filesystem timestamps."""

    if isinstance(value, Mapping):
        return {
            key: stable_identity_value(item)
            for key, item in value.items()
            if key != "mtime_ns"
        }
    if isinstance(value, list):
        return [stable_identity_value(item) for item in value]
    return value


def fusion_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select exactly the fields that define a v2 fusion artifact."""

    return {
        "kind": manifest.get("kind"),
        "schema_version": manifest.get("schema_version"),
        "identity_version": manifest.get("identity_version"),
        "parameters": manifest.get("parameters"),
        "upstream_artifact_ids": manifest.get("upstream_artifact_ids"),
        "inputs": stable_identity_value(manifest.get("inputs")),
        "outputs": stable_identity_value(manifest.get("outputs")),
    }


def fusion_artifact_id(manifest: Mapping[str, Any]) -> str:
    """Compute the canonical SHA-256 identity of a v2 fusion manifest."""

    encoded = json.dumps(
        fusion_identity_payload(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
