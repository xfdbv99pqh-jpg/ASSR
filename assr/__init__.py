# ASSR v1.5.0 - True Auto-Configuration
from .core import (
    ASSRTrainer, ASSRConfig, auto_calibrate,
    compute_stable_rank, compute_stable_rank_ratio,
    compute_condition_number, compute_spectral_health,
    print_spectral_report, analyze_layer, __version__,
)
__all__ = [
    "ASSRTrainer", "ASSRConfig", "auto_calibrate",
    "compute_stable_rank", "compute_stable_rank_ratio",
    "compute_condition_number", "compute_spectral_health",
    "print_spectral_report", "analyze_layer", "__version__",
]
