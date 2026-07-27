"""Atomic, lossless export and diagnostics for robust association."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import RobustAssociationConfig
from .types import AssociationEdge, InstanceTrack, MaskObservation, RefinedView


ALGORITHM_NAME = "robust_graph_spatial_v2"


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _dataset_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _palette(count: int) -> np.ndarray:
    colors = np.zeros((count + 1, 3), dtype=np.uint8)
    for index in range(1, count + 1):
        hue = (index * 0.618033988749895) % 1.0
        hsv = np.uint8([[[round(hue * 179), 220, 255]]])
        colors[index] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return colors


def _edge_arrays(
    edges: list[AssociationEdge],
    selected_edges: list[AssociationEdge],
) -> dict[str, np.ndarray]:
    selected_pairs = {
        (edge.source, edge.target, edge.source_view, edge.target_view)
        for edge in selected_edges
    }
    return {
        "source": np.asarray([edge.source for edge in edges], dtype=np.int32),
        "target": np.asarray([edge.target for edge in edges], dtype=np.int32),
        "source_view": np.asarray([edge.source_view for edge in edges], dtype=np.int16),
        "target_view": np.asarray([edge.target_view for edge in edges], dtype=np.int16),
        "score": np.asarray([edge.score for edge in edges], dtype=np.float32),
        "weighted_jaccard": np.asarray(
            [edge.weighted_jaccard for edge in edges], dtype=np.float32
        ),
        "coverage": np.asarray(
            [edge.bidirectional_coverage for edge in edges], dtype=np.float32
        ),
        "appearance": np.asarray([edge.appearance for edge in edges], dtype=np.float32),
        "spatial": np.asarray([edge.spatial for edge in edges], dtype=np.float32),
        "quality": np.asarray([edge.quality for edge in edges], dtype=np.float32),
        "selected": np.asarray(
            [
                (
                    edge.source,
                    edge.target,
                    edge.source_view,
                    edge.target_view,
                )
                in selected_pairs
                for edge in edges
            ],
            dtype=np.bool_,
        ),
    }


def _remap_labels(
    labels: np.ndarray,
    mapping: dict[int, int],
    ignore_label: int,
) -> np.ndarray:
    lookup = np.zeros(ignore_label + 1, dtype=np.uint16)
    for old_id, new_id in mapping.items():
        lookup[old_id] = new_id
    remapped = lookup[labels.astype(np.int64, copy=False)]
    remapped[labels == ignore_label] = ignore_label
    return remapped


def _qa_metrics(
    raw_labels: np.ndarray,
    labels: np.ndarray,
    ignore_label: int,
) -> dict[str, float | int]:
    foreground = raw_labels > 0
    foreground_pixels = int(foreground.sum())
    ignored = foreground & (labels == ignore_label)
    ignore_fraction = int(ignored.sum()) / max(foreground_pixels, 1)

    labeled = foreground & (labels > 0) & (labels != ignore_label)
    labeled_pixels = int(labeled.sum())
    if labeled_pixels:
        local_ids = raw_labels[labeled].astype(np.int64, copy=False)
        global_ids = labels[labeled].astype(np.int64, copy=False)
        stride = int(global_ids.max()) + 1
        pairs, counts = np.unique(local_ids * stride + global_ids, return_counts=True)
        pair_local_ids = pairs // stride
        dominant = np.zeros(int(pair_local_ids.max()) + 1, dtype=np.int64)
        np.maximum.at(dominant, pair_local_ids, counts)
        region_purity = float(dominant.sum() / labeled_pixels)
    else:
        region_purity = 0.0 if foreground_pixels else 1.0

    jump_count = 0
    comparable_count = 0
    for first_raw, second_raw, first_label, second_label in (
        (raw_labels[:, :-1], raw_labels[:, 1:], labels[:, :-1], labels[:, 1:]),
        (raw_labels[:-1, :], raw_labels[1:, :], labels[:-1, :], labels[1:, :]),
    ):
        comparable = (
            (first_raw == second_raw)
            & (first_raw > 0)
            & (first_label > 0)
            & (second_label > 0)
            & (first_label != ignore_label)
            & (second_label != ignore_label)
        )
        comparable_count += int(comparable.sum())
        jump_count += int((comparable & (first_label != second_label)).sum())
    return {
        "foreground_pixels": foreground_pixels,
        "ignore_pixels": int(ignored.sum()),
        "ignore_fraction": float(ignore_fraction),
        "region_purity": region_purity,
        "label_jump_count": jump_count,
        "label_jump_rate": float(jump_count / max(comparable_count, 1)),
    }


def export_association(
    *,
    scene_dir: Path,
    output_name: str,
    raw_mask_dir: Path,
    point_cloud: Path,
    refined_views: Iterable[RefinedView],
    observations: list[MaskObservation],
    tracks: list[InstanceTrack],
    candidate_edges: list[AssociationEdge],
    selected_edges: list[AssociationEdge],
    config: RobustAssociationConfig,
    visualize: bool,
    force: bool,
    extra_diagnostics: dict,
) -> tuple[Path, Path | None]:
    target = scene_dir / output_name
    declared_maximum = max((track.global_id for track in tracks), default=0)
    if declared_maximum >= config.ignore_label:
        raise ValueError(
            f"{declared_maximum} global IDs collide with ignore label "
            f"{config.ignore_label}"
        )
    staging = Path(tempfile.mkdtemp(prefix=f".{output_name}.staging-", dir=scene_dir))
    backup = None
    try:
        confidence_dir = staging / "confidence"
        valid_dir = staging / "valid"
        association_dir = staging / "association"
        confidence_dir.mkdir()
        valid_dir.mkdir()
        association_dir.mkdir()
        visualization_dir = staging / "visualization"
        if visualize:
            visualization_dir.mkdir()

        view_count = 0
        split_mask_count = 0
        uncertain_mask_count = 0
        ignore_pixel_count = 0
        label_paths = []
        for refined in refined_views:
            view_count += 1
            split_mask_count += int(refined.diagnostics["split_masks"])
            uncertain_mask_count += int(refined.diagnostics["uncertain_masks"])
            ignore_pixel_count += int(refined.diagnostics["ignore_pixels"])
            label_path = staging / f"{refined.image_name}.png"
            label_paths.append(label_path)
            if not cv2.imwrite(str(label_path), refined.labels.astype(np.uint16)):
                raise RuntimeError(f"Could not write label mask: {label_path}")
            confidence = np.round(refined.confidence * 255).astype(np.uint8)
            if not cv2.imwrite(
                str(confidence_dir / f"{refined.image_name}.png"),
                confidence,
            ):
                raise RuntimeError("Could not write confidence mask")
            if not cv2.imwrite(
                str(valid_dir / f"{refined.image_name}.png"),
                refined.valid.astype(np.uint8) * 255,
            ):
                raise RuntimeError("Could not write valid mask")

        active_old_ids: set[int] = set()
        for label_path in label_paths:
            labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            if labels is None or labels.dtype != np.uint16:
                raise RuntimeError(f"Could not rescan uint16 labels: {label_path}")
            active_old_ids.update(
                int(value)
                for value in np.unique(labels)
                if value not in (0, config.ignore_label)
            )
        mapping = {
            old_id: new_id
            for new_id, old_id in enumerate(sorted(active_old_ids), start=1)
        }
        maximum_label = len(mapping)
        if maximum_label >= config.ignore_label:
            raise ValueError(
                f"{maximum_label} compact IDs collide with ignore label "
                f"{config.ignore_label}"
            )

        colors = _palette(maximum_label)
        final_active_ids: set[int] = set()
        qa_views = []
        qa_failures = []
        for label_path in label_paths:
            labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            remapped = _remap_labels(labels, mapping, config.ignore_label)
            final_active_ids.update(
                int(value)
                for value in np.unique(remapped)
                if value not in (0, config.ignore_label)
            )
            if not cv2.imwrite(str(label_path), remapped):
                raise RuntimeError(f"Could not write compact labels: {label_path}")

            raw_path = raw_mask_dir / label_path.name
            raw_labels = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
            if raw_labels is None or raw_labels.ndim != 2:
                raise RuntimeError(f"Could not decode raw mask for QA: {raw_path}")
            if raw_labels.shape != remapped.shape:
                raise RuntimeError(f"Raw/output resolution differs for QA: {raw_path}")
            metrics = _qa_metrics(raw_labels, remapped, config.ignore_label)
            metrics["image_name"] = label_path.stem
            qa_views.append(metrics)
            violations = []
            if metrics["ignore_fraction"] > config.qa_max_ignore_fraction:
                violations.append("ignore_fraction")
            if metrics["region_purity"] < config.qa_min_region_purity:
                violations.append("region_purity")
            if metrics["label_jump_rate"] > config.qa_max_label_jump_rate:
                violations.append("label_jump_rate")
            if violations:
                qa_failures.append(
                    {"image_name": label_path.stem, "violations": violations, **metrics}
                )

            if visualize:
                safe_labels = remapped.astype(np.int64)
                valid = safe_labels != config.ignore_label
                safe_labels[~valid] = 0
                visualization = colors[safe_labels]
                visualization[~valid] = (127, 127, 127)
                if not cv2.imwrite(
                    str(visualization_dir / label_path.name),
                    visualization,
                ):
                    raise RuntimeError("Could not write visualization")

        expected_ids = set(range(1, maximum_label + 1))
        empty_classes = len(expected_ids - final_active_ids)
        if empty_classes:
            qa_failures.append(
                {"image_name": "<global>", "violations": ["empty_classes"]}
            )
        declared_ids = {track.global_id for track in tracks if track.global_id > 0}
        empty_before_compaction = len(declared_ids - active_old_ids)

        track_summaries = []
        for track in tracks:
            summary = track.summary()
            summary["global_id"] = mapping.get(track.global_id, 0)
            if track.global_id > 0 and summary["global_id"] == 0:
                summary["status"] = "pruned"
            track_summaries.append(summary)
        observation_summaries = []
        for observation in observations:
            summary = observation.summary()
            summary["track_id"] = mapping.get(observation.track_id, 0)
            if observation.track_id > 0 and summary["track_id"] == 0:
                summary["status"] = "pruned"
            observation_summaries.append(summary)

        qa_summary = {
            "passed": not qa_failures,
            "thresholds": {
                "empty_classes": 0,
                "max_ignore_fraction_per_frame": config.qa_max_ignore_fraction,
                "min_region_purity": config.qa_min_region_purity,
                "max_label_jump_rate": config.qa_max_label_jump_rate,
            },
            "empty_classes_before_compaction": empty_before_compaction,
            "empty_classes_after_compaction": empty_classes,
            "max_ignore_fraction": max(
                (item["ignore_fraction"] for item in qa_views), default=0.0
            ),
            "min_region_purity": min(
                (item["region_purity"] for item in qa_views), default=1.0
            ),
            "max_label_jump_rate": max(
                (item["label_jump_rate"] for item in qa_views), default=0.0
            ),
            "failed_views": len(qa_failures),
            "views": qa_views,
        }
        _write_json(association_dir / "qa.json", qa_summary)
        if qa_failures:
            preview = "; ".join(
                f"{item['image_name']}={','.join(item['violations'])}"
                for item in qa_failures[:8]
            )
            raise RuntimeError(
                f"Association QA gate failed for {len(qa_failures)} view(s): "
                f"max_ignore={qa_summary['max_ignore_fraction']:.4f}, "
                f"min_purity={qa_summary['min_region_purity']:.4f}, "
                f"max_jump={qa_summary['max_label_jump_rate']:.4f}; {preview}"
            )

        with (association_dir / "observations.jsonl").open(
            "w",
            encoding="utf-8",
        ) as stream:
            for summary in observation_summaries:
                stream.write(json.dumps(summary, sort_keys=True))
                stream.write("\n")
        _write_json(
            association_dir / "tracks.json",
            track_summaries,
        )
        np.savez_compressed(
            association_dir / "graph_edges.npz",
            **_edge_arrays(candidate_edges, selected_edges),
        )
        raw_paths = sorted(raw_mask_dir.glob("*.png"))
        diagnostics = {
            **extra_diagnostics,
            "algorithm": ALGORITHM_NAME,
            "views": view_count,
            "observations": len(observations),
            "candidate_edges": len(candidate_edges),
            "selected_edges": len(selected_edges),
            "tracks": len(tracks),
            "confirmed_tracks": sum(
                item["status"] == "confirmed" for item in track_summaries
            ),
            "propagated_tracks": sum(
                item["status"] == "propagated" for item in track_summaries
            ),
            "promoted_tracks": sum(
                item["status"] == "promoted" for item in track_summaries
            ),
            "tentative_tracks": sum(
                item["status"] == "tentative" for item in track_summaries
            ),
            "rejected_tracks": sum(
                item["status"] == "rejected" for item in track_summaries
            ),
            "pruned_tracks": sum(
                item["status"] == "pruned" for item in track_summaries
            ),
            "global_instances": maximum_label,
            "split_masks": split_mask_count,
            "uncertain_masks": uncertain_mask_count,
            "ignore_pixels": ignore_pixel_count,
            "empty_classes_before_compaction": empty_before_compaction,
            "qa": {key: value for key, value in qa_summary.items() if key != "views"},
        }
        _write_json(association_dir / "diagnostics.json", diagnostics)
        manifest = {
            "format_version": 1,
            "complete": True,
            "algorithm": ALGORITHM_NAME,
            "output_name": output_name,
            "label_dtype": "uint16",
            "background_label": 0,
            "ignore_label": config.ignore_label,
            "num_mask": maximum_label,
            "point_cloud": str(point_cloud.resolve()),
            "point_cloud_size": point_cloud.stat().st_size,
            "raw_mask_folder": str(raw_mask_dir.resolve()),
            "raw_mask_fingerprint": _dataset_fingerprint(raw_paths),
            "config": config.to_dict(),
            "diagnostics": diagnostics,
        }
        _write_json(association_dir / "manifest.json", manifest)
        # Legacy readers consume info.json and num_mask.
        _write_json(
            staging / "info.json",
            {
                "num_mask": maximum_label,
                "raw_mask_folder": str(raw_mask_dir.resolve()),
                "associated_mask_folder": str(target.resolve()),
                "front_percentage": config.front_percentage,
                "iou_threshold": config.match_threshold,
                "num_patch": config.num_patches,
                "algorithm": ALGORITHM_NAME,
                "label_dtype": "uint16",
                "ignore_label": manifest["ignore_label"],
                "confidence_folder": str((target / "confidence").resolve()),
                "valid_folder": str((target / "valid").resolve()),
                "association_manifest": str(
                    (target / "association" / "manifest.json").resolve()
                ),
            },
        )

        if target.exists():
            if not force:
                raise FileExistsError(
                    f"Associated-mask directory already exists: {target}. "
                    "Use --force or choose --output-name."
                )
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = target.with_name(f"{target.name}.backup-{timestamp}")
            suffix = 1
            while backup.exists():
                backup = target.with_name(f"{target.name}.backup-{timestamp}-{suffix}")
                suffix += 1
            target.rename(backup)
        try:
            os.replace(staging, target)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                backup.rename(target)
                backup = None
            raise
        return target, backup
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
