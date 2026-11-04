#!/usr/bin/env python3
"""Annotate removal holes from paired PGSR renders and propagate with DeAOT.

Uses only the selected-object reconstruction and virtual training views. No test
RGB/unseen_mask or another method's masks are read. Geometry checks detect drift;
an automatic correction is recorded and becomes a new DeAOT reference frame.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from run_geometry_ablation import read, write
from tracker_web import Workspace, overlay

REPO = Path(__file__).resolve().parents[2]


def hole_reference(original, removed, *, allow_empty=False):
    delta = np.max(np.abs(np.asarray(original) - np.asarray(removed)), axis=-1)
    if delta.ndim != 2 or not np.isfinite(delta).all():
        raise ValueError("invalid paired PGSR RGB render")
    mask = delta > max(2 / 255, float(delta.max()) * .02)
    mask = ndimage.binary_closing(mask, iterations=2)
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max(20, mask.size * .00002)
    keep[0] = False
    mask = ndimage.binary_fill_holes(keep[labels])
    if allow_empty and not mask.any():
        return mask
    if not (0 if allow_empty else 50) <= int(mask.sum()) < mask.size * .65:
        raise ValueError(f"unusable removal footprint: {int(mask.sum())}/{mask.size}")
    return mask


def correct_prediction(prediction, reference):
    prediction = np.asarray(prediction) != 0
    if prediction.shape != reference.shape:
        raise ValueError("tracking dimensions differ from removal footprint")
    if not reference.any():
        return reference.copy(), dict(reference_coverage=1., support_precision=1.,
                                     reset_reference=True, area=0, reference_area=0)
    support = ndimage.binary_dilation(reference, iterations=12)
    recall = float((prediction & reference).sum() / reference.sum())
    precision = float((prediction & support).sum() / max(1, prediction.sum()))
    reset = recall < .85 or precision < .6 or prediction.sum() > reference.size * .65
    # A corrected annotation must cover the known removed geometry. A narrow
    # support band permits temporal boundary refinement without table-wide drift.
    corrected = reference.copy() if reset else (prediction & support) | reference
    return corrected, dict(reference_coverage=recall, support_precision=precision,
                          reset_reference=bool(reset), area=int(corrected.sum()),
                          reference_area=int(reference.sum()))


class GuidedDeAOT:
    def __init__(self, references):
        self.references = references
        self.statistics = []

    def propagate(self, frames, mask):
        import torch
        from aot_tracker import get_aot
        from model_args import aot_args
        tracker = get_aot(copy.deepcopy(aot_args))
        tracker.restart()
        with torch.inference_mode(), torch.cuda.amp.autocast():
            tracker.add_reference_frame(frames[0], mask, 1, 0)
            self.statistics.append(dict(frame=0, reference_coverage=1., support_precision=1.,
                reset_reference=False, area=int(mask.sum()), reference_area=int(mask.sum())))
            yield mask
            for i, frame in enumerate(frames[1:], 1):
                predicted = tracker.track(frame)
                current, stats = correct_prediction(predicted.squeeze().cpu().numpy(), self.references[i])
                if stats["reset_reference"]:
                    tracker.restart()
                    tracker.add_reference_frame(frame, current.astype(np.uint8), 1, 0)
                else:
                    memory = torch.as_tensor(current, device=predicted.device,
                                             dtype=predicted.dtype)[None, None]
                    tracker.update_memory(memory)
                self.statistics.append(dict(frame=i, **stats))
                if i % 10 == 0 or stats["reset_reference"]:
                    print(f"DeAOT {i + 1}/{len(frames)} recall={stats['reference_coverage']:.3f} "
                          f"precision={stats['support_precision']:.3f} correction={stats['reset_reference']}", flush=True)
                yield current.astype(np.uint8)


class OcclusionGuidedDeAOT(GuidedDeAOT):
    """Restart memory at reappearance; never invent a hole in an occluded frame."""
    def propagate(self, frames, mask):
        import torch
        from aot_tracker import get_aot
        from model_args import aot_args
        tracker = get_aot(copy.deepcopy(aot_args))
        active = False
        with torch.inference_mode(), torch.cuda.amp.autocast():
            for i, frame in enumerate(frames):
                ref = self.references[i]
                if not ref.any() or not active:
                    current = ref.copy()
                    stats = dict(reference_coverage=1., support_precision=1., reset_reference=True,
                                 area=int(ref.sum()), reference_area=int(ref.sum()))
                else:
                    predicted = tracker.track(frame)
                    current, stats = correct_prediction(predicted.squeeze().cpu().numpy(), ref)
                if not current.any():
                    tracker.restart(); active = False
                elif stats['reset_reference']:
                    tracker.restart()
                    tracker.add_reference_frame(frame, current.astype(np.uint8), 1, 0)
                    active = True
                else:
                    memory = torch.as_tensor(current, device=predicted.device, dtype=predicted.dtype)[None, None]
                    tracker.update_memory(memory)
                self.statistics.append(dict(frame=i, **stats))
                print(f"DeAOT {i+1}/{len(frames)} area={stats['area']} reset={stats['reset_reference']}", flush=True)
                yield current.astype(np.uint8)


def review_pages(frames, masks, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for first in range(0, len(frames), 12):
        page = Image.new("RGB", (1280, 4 * 255), "white")
        draw = ImageDraw.Draw(page)
        for j, index in enumerate(range(first, min(first + 12, len(frames)))):
            x, y = (j % 3) * 426, (j // 3) * 255
            im = Image.fromarray(overlay(frames[index], masks[index])); im.thumbnail((420, 230))
            page.paste(im, (x, y + 20)); draw.text((x + 5, y + 3), f"frame {index:05d}", fill="black")
        page.save(destination / f"frames_{first:05d}_{min(first + 11, len(frames)-1):05d}.jpg", quality=88)


def run(args):
    removal = args.removal_root.resolve(strict=True)
    tracker_root = REPO / "submodules/Inpaint360GS/Segment-and-Track-Anything"
    sys.path.insert(0, str(tracker_root)); os.chdir(tracker_root)
    from virtual_render_io import validate_render, sha256
    original_root = removal / "work_model/virtual/ours_2000"
    removed_root = removal / "work_model/virtual/ours_object_removal/iteration_2000"
    original = validate_render(original_root); removed = validate_render(removed_root)
    if original["camera_artifact_id"] != removed["camera_artifact_id"]:
        raise ValueError("paired render cameras differ")
    names = sorted(original["frames"])
    references = [hole_reference(np.load(original_root / "rgb_raw" / f"{n}.npy"),
                                 np.load(removed_root / "rgb_raw" / f"{n}.npy"),
                                 allow_empty=args.allow_empty_masks) for n in names]
    if not any(r.any() for r in references):
        raise ValueError("target is unobservable in the entire camera path")
    models = (OcclusionGuidedDeAOT if args.allow_empty_masks else GuidedDeAOT)(references)
    workspace = Workspace(removal / "tracker/images.zip", removal / "tracker/results", models,
                          allow_empty_masks=args.allow_empty_masks)
    if [Path(n).stem for n in workspace.names] != names:
        raise ValueError("paired render/archive frame order differs")
    if workspace.destination.exists():
        raise ValueError("existing masks are not overwritten")
    workspace.mask = references[0].astype(np.uint8)
    workspace.destination.parent.mkdir(parents=True, exist_ok=True)
    workspace.busy.acquire()
    workspace._track()  # Same atomic output/active-session checks as the browser.
    if workspace.phase != "done":
        raise RuntimeError(workspace.message)
    review_pages(workspace.frames, workspace.masks, removal / "tracker/review")
    write(removal / "tracker/automatic_annotation.json", dict(
        complete=True, method="PGSR original-minus-removed footprint + DeAOT with recorded geometric corrections",
        test_data_used=False, visually_reviewed_all_frames=False,
        original_render_sha256=sha256(original_root / "render_manifest.json"),
        removed_render_sha256=sha256(removed_root / "render_manifest.json"),
        frame_count=len(names), corrected_frames=[s["frame"] for s in models.statistics if s["reset_reference"]],
        statistics=models.statistics, review_directory=str(removal / "tracker/review")))
    print(f"Headless tracking complete: {workspace.destination}; commit through run_remove Stage 5", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--removal-root", type=Path, required=True)
    parser.add_argument("--allow-empty-masks", action="store_true")
    run(parser.parse_args())
