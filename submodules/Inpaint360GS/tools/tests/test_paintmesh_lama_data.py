from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.prepare_paintmesh_lama_data import (
    LamaDataError,
    prepare_lama_inputs,
    validate_lama_outputs,
)


class LamaFixture:
    def __init__(self, root: Path, frames: int = 2) -> None:
        self.root = root
        self.frames = frames
        self.masks = root / "tracker_masks"
        self.rgb = root / "removed_rgb"
        self.depth = root / "removed_depth"
        self.reference = root / "reference_depth"
        self.color_input = root / "run" / "input" / "color"
        self.depth_input = root / "run" / "input" / "depth"
        self.color_output = root / "run" / "output" / "color"
        self.depth_output = root / "run" / "output" / "depth"
        self.input_manifest = root / "run" / "input_manifest.json"
        self.completion_manifest = root / "run" / "completion_manifest.json"
        self.model = root / "model"
        for directory in (self.masks, self.rgb, self.depth, self.reference):
            directory.mkdir(parents=True)
        (self.model / "models").mkdir(parents=True)
        (self.model / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (self.model / "models" / "best.ckpt").write_bytes(b"checkpoint")

        for index in range(frames):
            stem = f"{index:05d}"
            image = np.zeros((8, 9, 3), dtype=np.uint8)
            image[..., 0] = 10 + index
            image[..., 1] = np.arange(9, dtype=np.uint8)[None, :]
            Image.fromarray(image, mode="RGB").save(self.rgb / f"{stem}.png")

            # A palette entry with a dark RGB value would be discarded by the
            # historical grayscale >=128 conversion.  The adapter must use the
            # original index value, for which this region is identity 1.
            labels = np.zeros((8, 9), dtype=np.uint8)
            labels[3:5, 4:6] = 1
            mask = Image.fromarray(labels, mode="P")
            palette = [0] * 768
            palette[3:6] = [1, 1, 1]
            mask.putpalette(palette)
            mask.save(self.masks / f"{stem}.png")

            yy, xx = np.mgrid[:8, :9]
            removed = (1.0 + xx + yy * 0.1 + index).astype(np.float32)
            reference = (2.0 + xx * 0.5 + yy + index).astype(np.float32)
            np.save(self.depth / f"{stem}.npy", removed)
            np.save(self.reference / f"{stem}.npy", reference)

    def prepare(self):
        return prepare_lama_inputs(
            self.masks,
            self.rgb,
            self.depth,
            self.reference,
            self.color_input,
            self.depth_input,
            self.input_manifest,
            frames=self.frames,
            min_area=3,
            dilation=1,
        )

    def make_valid_outputs(self) -> None:
        self.color_output.mkdir(parents=True)
        self.depth_output.mkdir(parents=True)
        for index in range(self.frames):
            stem = f"{index:05d}"
            rgb = np.asarray(
                Image.open(self.color_input / f"{stem}.png").convert("RGB")
            ).copy()
            mask = np.asarray(Image.open(self.color_input / f"{stem}_mask.png")) != 0
            rgb[mask] = [200, 100, 50]
            Image.fromarray(rgb, mode="RGB").save(self.color_output / f"{stem}.png")

            depth = np.load(self.depth_input / f"{stem}.npy")
            depth = depth.copy()
            depth[mask] += 0.25
            np.save(self.depth_output / f"{stem}.npy", depth)


class PreparePaintMeshLamaDataTest(unittest.TestCase):
    def test_palette_indices_are_binarized_and_manifest_is_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LamaFixture(Path(temporary))
            payload = fixture.prepare()

            self.assertTrue(payload["complete"])
            self.assertEqual(payload["kind"], "paintmesh-lama-inputs")
            mask = np.asarray(Image.open(fixture.color_input / "00000_mask.png"))
            self.assertGreater(int((mask != 0).sum()), 4)
            self.assertEqual(set(np.unique(mask)), {0, 255})
            self.assertEqual(
                np.load(fixture.depth_input / "00000.npy").shape,
                (8, 9),
            )
            reused = fixture.prepare()
            self.assertEqual(reused["artifact_id"], payload["artifact_id"])

            on_disk = json.loads(fixture.input_manifest.read_text(encoding="utf-8"))
            self.assertTrue(on_disk["complete"])
            self.assertEqual(len(on_disk["frames"]), fixture.frames)

    def test_exact_frame_set_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LamaFixture(Path(temporary))
            Image.new("L", (9, 8), color=255).save(fixture.rgb / "00002.png")
            with self.assertRaisesRegex(LamaDataError, "exactly 2 canonical frames"):
                fixture.prepare()

    def test_depth_must_be_finite_and_have_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LamaFixture(Path(temporary))
            broken = np.ones((8, 9), dtype=np.float32)
            broken[0, 0] = np.nan
            np.save(fixture.depth / "00000.npy", broken)
            with self.assertRaisesRegex(LamaDataError, "NaN or infinity"):
                fixture.prepare()

    def test_shape_mismatch_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LamaFixture(Path(temporary))
            np.save(
                fixture.reference / "00000.npy",
                np.arange(16, dtype=np.float32).reshape(4, 4),
            )
            with self.assertRaisesRegex(LamaDataError, "does not match RGB"):
                fixture.prepare()
            self.assertFalse(fixture.color_input.exists())

    def test_completion_preserves_outside_mask_and_is_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LamaFixture(Path(temporary))
            fixture.prepare()
            fixture.make_valid_outputs()
            payload = validate_lama_outputs(
                fixture.color_input,
                fixture.depth_input,
                fixture.color_output,
                fixture.depth_output,
                fixture.model,
                fixture.input_manifest,
                fixture.completion_manifest,
                frames=fixture.frames,
            )
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["kind"], "paintmesh-lama-completion")
            self.assertEqual(len(payload["frames"]), fixture.frames)

            rgb_path = fixture.color_output / "00000.png"
            rgb = np.asarray(Image.open(rgb_path).convert("RGB")).copy()
            rgb[0, 0] = [255, 255, 255]
            Image.fromarray(rgb, mode="RGB").save(rgb_path)
            fixture.completion_manifest.unlink()
            with self.assertRaisesRegex(LamaDataError, "outside the mask"):
                validate_lama_outputs(
                    fixture.color_input,
                    fixture.depth_input,
                    fixture.color_output,
                    fixture.depth_output,
                    fixture.model,
                    fixture.input_manifest,
                    fixture.completion_manifest,
                    frames=fixture.frames,
                )


if __name__ == "__main__":
    unittest.main()
