# =============================================================================
# ASSR: Auto-Calibrated Stochastic Spectral Regularization
# Version: 1.4.0 - Biopsy Mode
# =============================================================================
#
# Usage:
#   from assr import ASSRTrainer, auto_calibrate
#   config = auto_calibrate(model)  # Auto-calibrates with Biopsy Mode
#   trainer = ASSRTrainer(model=model, args=args, assr_config=config, ...)
#   trainer.train()
#   print(trainer.assr_stats)
#
# =============================================================================

from .core import (
    # Main interface
    ASSRTrainer,
    ASSRConfig,
    auto_calibrate,
    
    # Spectral metrics
    compute_stable_rank,
    compute_stable_rank_ratio,
    compute_condition_number,
    compute_spectral_health,
    
    # Utilities
    print_spectral_report,
    analyze_layer,
    
    # Version
    __version__,
)

__all__ = [
    "ASSRTrainer",
    "ASSRConfig",
    "auto_calibrate",
    "compute_stable_rank",
    "compute_stable_rank_ratio",
    "compute_condition_number",
    "compute_spectral_health",
    "print_spectral_report",
    "analyze_layer",
    "__version__",
]
