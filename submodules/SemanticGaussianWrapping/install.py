"""Build the CUDA rasterizer and validate the Python environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dependencies", action="store_true")
    parser.add_argument("--skip-extension", action="store_true")
    parser.add_argument(
        "--cuda-arch-list",
        help=(
            "semicolon-separated CUDA compute capabilities, for example "
            "'8.6' or '7.5;8.0;8.6+PTX'; overrides TORCH_CUDA_ARCH_LIST"
        ),
    )
    parser.add_argument("--editable", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.cuda_arch_list is not None and not args.cuda_arch_list.strip():
        parser.error("--cuda-arch-list cannot be empty")

    if not args.skip_dependencies:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    if not args.skip_extension:
        extension = ROOT / "submodules" / "diff-semantic-gaussian-rasterization"
        if not extension.exists():
            raise FileNotFoundError(f"CUDA extension source is missing: {extension}")
        build_env = os.environ.copy()
        if args.cuda_arch_list is not None:
            build_env["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch_list.strip()
        run(
            [sys.executable, "-m", "pip", "install", "-e", str(extension), "--no-build-isolation"],
            env=build_env,
        )
    if args.editable:
        run([sys.executable, "-m", "pip", "install", "-e", str(ROOT), "--no-deps"])


if __name__ == "__main__":
    main()
