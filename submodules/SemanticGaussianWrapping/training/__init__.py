"""Training engine and experiment state."""

from .engine import SemanticGaussianTrainer, TrainingResult
from .checkpointing import (
    atomic_torch_save,
    capture_rng_state,
    resolve_training_configuration,
    restore_rng_state,
    validate_training_checkpoint_header,
    validate_resume_source,
)

__all__ = [
    "SemanticGaussianTrainer",
    "TrainingResult",
    "atomic_torch_save",
    "capture_rng_state",
    "resolve_training_configuration",
    "restore_rng_state",
    "validate_training_checkpoint_header",
    "validate_resume_source",
]
