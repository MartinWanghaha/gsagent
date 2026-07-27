from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh import (  # noqa: E402
    AdaptiveSamplingConfig,
    BlockedGridSampler,
    Bounds,
    SurfaceFieldAdapter,
    marching_cubes_blocks,
)


class _DetailedHalfSphere:
    device = "cpu"
    query_chunk_size = 100_000

    def query(self, points, chunk_size=None):
        points = np.asarray(points, dtype=np.float32)
        radius = np.linalg.norm(points, axis=1)
        sdf = radius - 0.8
        normal = points / np.maximum(radius[:, None], 1e-8)
        posterior = np.zeros((len(points), 5), dtype=np.float32)
        posterior[:, 0] = 1.0
        detailed = points[:, 0] < 0
        posterior[detailed, 0] = 0.0
        posterior[detailed, 2] = 1.0
        return {
            "occupancy": (1.0 / (1.0 + np.exp(20.0 * sdf))).astype(np.float32),
            "sdf": sdf.astype(np.float32),
            "normal": normal.astype(np.float32),
            "semantic": np.tile(
                np.asarray([[1.0, 0.0]], dtype=np.float32), (len(points), 1)
            ),
            "geometry_posterior": posterior,
            "uncertainty": np.zeros(len(points), dtype=np.float32),
        }


class _SingleActiveProbe:
    device = "cpu"

    def query(self, points, chunk_size=None):
        points = np.asarray(points, dtype=np.float32)
        active = np.linalg.norm(points, axis=1) < 1e-7
        sdf = np.where(active, 0.0, 10.0).astype(np.float32)
        count = len(points)
        return {
            "occupancy": np.where(active, 0.5, 0.0).astype(np.float32),
            "sdf": sdf,
            "normal": np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
            "semantic": np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (count, 1)),
            "geometry_posterior": np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (count, 1),
            ),
            "uncertainty": np.zeros(count, dtype=np.float32),
        }


def test_touching_adaptive_blocks_use_conforming_boundary_lattices() -> None:
    geometry_field = SurfaceFieldAdapter(
        _DetailedHalfSphere(), input_type="numpy", decode_semantics=False
    )
    sampler = BlockedGridSampler(
        geometry_field,
        Bounds(np.full(3, -1.2), np.full(3, 1.2)),
        blocks_per_axis=2,
        block_cells=4,
        max_refinement=2,
        config=AdaptiveSamplingConfig(detail_posterior_threshold=0.35),
    )
    blocks = sampler.sample_blocks()
    assert len({block.refinement_level for block in blocks}) == 1

    mesh = marching_cubes_blocks(blocks, sampler.field, value="sdf", level=0.0)
    edges = np.sort(
        mesh.faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert not np.any(counts == 1)


def test_active_region_includes_one_face_neighbor_halo() -> None:
    sampler = BlockedGridSampler(
        _SingleActiveProbe(),
        Bounds(np.full(3, -1.0), np.full(3, 1.0)),
        blocks_per_axis=3,
        block_cells=1,
        max_refinement=1,
    )
    blocks = sampler.sample_blocks()
    assert sum(block.decision.active for block in blocks) == 1
    assert len(blocks) == 7
    assert sampler.last_halo_blocks == 6
    assert {block.refinement_level for block in blocks} == {1}
