"""Command-line parameter groups following the original 3DGS interface."""

from __future__ import annotations

import ast
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Any


class GroupParams:
    """Small namespace used by :class:`ParamGroup.extract`."""


class ParamGroup:
    """Register public attributes as arguments on an ``ArgumentParser``.

    A leading underscore requests the conventional one-letter alias (for
    example ``_source_path`` becomes ``--source_path``/``-s``).  The instance
    keeps the defaults, so ``extract`` can be used with a parser that contains
    arguments from several groups.
    """

    def __init__(
        self,
        parser: ArgumentParser,
        name: str,
        fill_none: bool = False,
    ) -> None:
        group = parser.add_argument_group(name)
        for stored_name, default in vars(self).items():
            shorthand = stored_name.startswith("_")
            name = stored_name[1:] if shorthand else stored_name
            value = None if fill_none else default
            flags = [f"--{name}"]
            if shorthand:
                flags.append(f"-{name[0]}")
            if isinstance(default, bool):
                action = "store_false" if default else "store_true"
                group.add_argument(*flags, default=value, action=action)
            elif isinstance(default, (tuple, list)):
                element_type = type(default[0]) if default else str
                group.add_argument(
                    *flags,
                    default=value,
                    nargs=len(default) if isinstance(default, tuple) else "+",
                    type=element_type,
                )
            else:
                group.add_argument(*flags, default=value, type=type(default))

    def extract(self, args: Namespace) -> Namespace:
        selected: dict[str, Any] = {}
        own = vars(self)
        for name, value in vars(args).items():
            if name in own or f"_{name}" in own:
                selected[name] = value
        return Namespace(**selected)


class ModelParams(ParamGroup):
    def __init__(self, parser: ArgumentParser, sentinel: bool = False) -> None:
        self.sh_degree = 3
        self.semantic_dim = 16
        self.geometry_experts = 5
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.llffhold = 8
        self.random_points = 100_000
        self.load_device = "cpu"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args: Namespace) -> Namespace:
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        group.model_path = os.path.abspath(group.model_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser: ArgumentParser) -> None:
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.reference_rasterizer = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser: ArgumentParser) -> None:
        self.iterations = 30_000
        self.position_lr_init = 1.6e-4
        self.position_lr_final = 1.6e-6
        self.position_lr_delay_steps = 0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 2.5e-3
        self.opacity_lr = 5.0e-2
        self.scaling_lr = 5.0e-3
        self.rotation_lr = 1.0e-3
        self.semantic_lr = 2.5e-3
        self.geometry_lr = 1.0e-3
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_semantic = 0.1
        self.lambda_geometry = 0.05
        self.lambda_surface = 0.05
        self.lambda_mesh = 0.01
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")


class SemanticParams(ParamGroup):
    def __init__(self, parser: ArgumentParser) -> None:
        self.semantic_path = "sam_mask"
        self.semantic_confidence_path = ""
        self.semantic_boundary_path = ""
        self.semantic_ignore_label = -1
        self.semantic_background_label = 0
        self.semantic_temperature = 0.1
        self.semantic_start = 3_000
        self.joint_geometry_start = 7_000
        self.surface_refine_start = 20_000
        self.semantic_confidence_floor = 0.05
        self.geometry_confidence_threshold = 0.35
        self.boundary_width = 2
        self.gaga_info_file = "info.json"
        super().__init__(parser, "Semantic Parameters")


class DensityParams(ParamGroup):
    def __init__(self, parser: ArgumentParser) -> None:
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densification_interval = 100
        self.opacity_reset_interval = 3_000
        self.densify_grad_threshold = 2.0e-4
        self.geometry_error_threshold = 0.05
        self.boundary_split_boost = 1.5
        self.small_instance_protection = 0.8
        self.min_opacity = 0.005
        self.max_screen_size = 20.0
        self.split_children = 2
        self.max_gaussians = 5_000_000
        super().__init__(parser, "Unified Density Parameters")


def get_combined_args(parser: ArgumentParser) -> Namespace:
    """Merge command-line options with the standard ``cfg_args`` snapshot.

    ``ast.literal_eval`` is intentionally used instead of the original
    unrestricted ``eval``.  Both ``Namespace(foo=...)`` and a plain dict are
    accepted.
    """

    command_line = parser.parse_args(sys.argv[1:])
    model_path = getattr(command_line, "model_path", None)
    config_values: dict[str, Any] = {}
    if model_path:
        config_path = os.path.join(model_path, "cfg_args")
        if os.path.isfile(config_path):
            text = open(config_path, "r", encoding="utf8").read().strip()
            if text.startswith("Namespace(") and text.endswith(")"):
                expression = ast.parse(text, mode="eval").body
                if (
                    not isinstance(expression, ast.Call)
                    or not isinstance(expression.func, ast.Name)
                    or expression.func.id != "Namespace"
                    or expression.args
                    or any(keyword.arg is None for keyword in expression.keywords)
                ):
                    raise ValueError(f"invalid cfg_args expression in {config_path}")
                parsed = {
                    keyword.arg: ast.literal_eval(keyword.value)
                    for keyword in expression.keywords
                }
            else:
                parsed = ast.literal_eval(text)
            config_values = vars(parsed) if isinstance(parsed, Namespace) else dict(parsed)

    merged = dict(config_values)
    merged.update({k: v for k, v in vars(command_line).items() if v is not None})
    return Namespace(**merged)


__all__ = [
    "DensityParams",
    "GroupParams",
    "ModelParams",
    "OptimizationParams",
    "ParamGroup",
    "PipelineParams",
    "SemanticParams",
    "get_combined_args",
]
