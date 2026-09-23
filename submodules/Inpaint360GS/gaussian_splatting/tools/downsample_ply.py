#!/usr/bin/env python3
# pip install plyfile
import os
import numpy as np
from plyfile import PlyData, PlyElement
from typing import Optional

def downsample_ply(
    input_path: str,
    downsample_ratio: float = 0.1,
    seed: Optional[int] = 0,
    output_path: Optional[str] = None
):
    assert 0 < downsample_ratio <= 1.0, "downsample_ratio  (0,1]"

    ply = PlyData.read(input_path)
    vert_elem = ply['vertex']         # PlyElement
    v = vert_elem.data                # numpy structured array (N,)

    names = list(v.dtype.names)
    for req in ('x', 'y', 'z'):
        if req not in names:
            raise ValueError(f" PLY  '{req}'")

    N = v.shape[0]
    k = max(1, int(round(N * downsample_ratio)))
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    idx = rng.choice(N, size=k, replace=False)

    dtype_out = [(n, v.dtype.fields[n][0]) for n in names]
    out = np.empty(k, dtype=dtype_out)

    for n in names:
        arr = v[n][idx]
        if n in ('x', 'y', 'z', 'nx', 'ny', 'nz'):
            out[n] = arr.astype(np.float32, copy=False)
        elif n in ('red', 'green', 'blue'):
            out[n] = arr.astype(np.uint8, copy=False)
        else:
            out[n] = arr

    if output_path is None:
        out_dir = os.path.dirname(input_path)
        out_name = f"points3D_downsample_{downsample_ratio:.1f}.ply"
        output_path = os.path.join(out_dir, out_name)

    PlyData([PlyElement.describe(out, 'vertex')], text=True).write(output_path)
    print(f"✅ Saved: {output_path}  (kept {k}/{N} = {100*downsample_ratio:.1f}%)")

if __name__ == "__main__":
    in_path = "./data/mipnerf360/garden_3/sparse/0/points3D.ply"
    downsample_ply(in_path, downsample_ratio=0.1, seed=0)
