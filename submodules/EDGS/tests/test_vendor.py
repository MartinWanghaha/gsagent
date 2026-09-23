from __future__ import annotations

import sys

from source.vendor import (
    DINOV2_ROOT,
    GAUSSIAN_SPLATTING_ROOT,
    MVROMA_ROOT,
    UFM_ROOT,
    UNICEPTION_ROOT,
    bootstrap_gaussian_splatting,
    bootstrap_mvroma,
)


def test_bootstrap_repositions_existing_vendor_path(monkeypatch):
    vendor_path = str(GAUSSIAN_SPLATTING_ROOT.resolve())
    monkeypatch.setattr(sys, "path", ["competitor", vendor_path, "tail", vendor_path])

    bootstrap_gaussian_splatting()

    assert sys.path == [vendor_path, "competitor", "tail"]


def test_mvroma_bootstrap_exposes_model_and_complete_prematcher(monkeypatch):
    monkeypatch.setattr(sys, "path", ["tail"])

    bootstrap_mvroma()

    assert sys.path[:3] == [
        str(MVROMA_ROOT.resolve()),
        str(UFM_ROOT.resolve()),
        str(UNICEPTION_ROOT.resolve()),
    ]
    assert sys.path[-1] == "tail"


def test_dinov2_vendor_path_is_owned_by_edgs():
    assert DINOV2_ROOT == MVROMA_ROOT.parent / "DINOv2"
