# =============================================================================
# ASSR: Auto-Calibrated Stochastic Spectral Regularization
# =============================================================================
#
# Stabilizes LLM/ViT training by monitoring spectral health in real-time.
#
# Quick Start:
#   from assr import ASSRTrainer
#   trainer = ASSRTrainer(model=model, args=args, train_dataset=dataset)
#   trainer.train()
#
# =============================================================================

from .core import (
    # Main interface
    ASSRTrainer,
    ASSRConfig,
    
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
    "compute_stable_rank",
    "compute_stable_rank_ratio", 
    "compute_condition_number",
    "compute_spectral_health",
    "print_spectral_report",
    "analyze_layer",
    "__version__",
]
