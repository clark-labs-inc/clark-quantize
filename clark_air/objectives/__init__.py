"""Teacher-behavior objectives for Clark Air."""

from .codebook_geometry import codebook_attraction_loss, threshold_margin_loss
from .diffusion_teacher_velocity import symmetric_velocity_loss, velocity_loss
from .hidden_geometry import centered_cka_loss, relational_gram_loss
from .llm_teacher_kl import teacher_kl_loss

__all__ = [
    "centered_cka_loss",
    "codebook_attraction_loss",
    "relational_gram_loss",
    "symmetric_velocity_loss",
    "teacher_kl_loss",
    "threshold_margin_loss",
    "velocity_loss",
]
