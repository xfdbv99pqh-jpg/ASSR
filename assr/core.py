# =============================================================================
# ASSR: Auto-Calibrated Stochastic Spectral Regularization
# Version: 1.4.0
# =============================================================================
#
# New in v1.4: Biopsy Mode - subsample large weight matrices for faster
# spectral computation on billion-parameter models.
#
# Usage:
#   from assr import ASSRTrainer, auto_calibrate
#   config = auto_calibrate(model)
#   trainer = ASSRTrainer(model=model, args=args, assr_config=config, ...)
#
# =============================================================================

import torch
import torch.nn as nn
import random
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

__version__ = "1.4.0"
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
]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ASSRConfig:
    """
    Configuration for ASSR regularization.
    
    Attributes:
        base_lambda: Base regularization strength. Default: 1e-4
        stable_rank_floor: Trigger if SR ratio below this. Default: 0.25
        condition_ceiling: Trigger if condition above this. Default: 500
        sample_ratio: Fraction of layers to check per step. Default: 0.1
        sample_freq: Check every N steps. Default: 1
        max_severity_multiplier: Max scaling for adaptive lambda. Default: 10.0
        subsample_limit: Max matrix dimension for SVD (Biopsy Mode). Default: 1024
            - Matrices larger than this are randomly subsampled
            - Set to None to disable (full SVD always)
            - Lower values = faster but less accurate
        log_interventions: Print debug info. Default: False
    """
    
    base_lambda: float = 1e-4
    stable_rank_floor: float = 0.25
    condition_ceiling: float = 500.0
    sample_ratio: float = 0.1
    sample_freq: int = 1
    max_severity_multiplier: float = 10.0
    subsample_limit: Optional[int] = 1024  # Biopsy Mode: subsample large matrices
    log_interventions: bool = False
    
    def __post_init__(self):
        assert 0 < self.sample_ratio <= 1.0, "sample_ratio must be in (0, 1]"
        assert self.base_lambda >= 0, "base_lambda must be non-negative"
        assert 0 < self.stable_rank_floor < 1, "stable_rank_floor must be in (0, 1)"
        assert self.condition_ceiling > 1, "condition_ceiling must be > 1"
        assert self.sample_freq >= 1, "sample_freq must be >= 1"
        if self.subsample_limit is not None:
            assert self.subsample_limit >= 64, "subsample_limit must be >= 64"


# =============================================================================
# BIOPSY MODE - MATRIX SUBSAMPLING
# =============================================================================

def _subsample_matrix(W: torch.Tensor, limit: int) -> torch.Tensor:
    """
    Subsample a large matrix for faster spectral computation.
    
    For a matrix larger than limit x limit, randomly select rows and columns
    to create a smaller representative sample. The spectral properties of
    the subsample approximate those of the full matrix.
    
    Args:
        W: Weight matrix of shape (m, n)
        limit: Maximum dimension size
        
    Returns:
        Subsampled matrix of shape (min(m, limit), min(n, limit))
    """
    m, n = W.shape
    
    if m <= limit and n <= limit:
        return W
    
    with torch.no_grad():
        # Random row indices
        if m > limit:
            row_idx = torch.randperm(m, device=W.device)[:limit]
            W = W[row_idx, :]
        
        # Random column indices
        if n > limit:
            col_idx = torch.randperm(n, device=W.device)[:limit]
            W = W[:, col_idx]
    
    return W


# =============================================================================
# SPECTRAL SENSORS (with Biopsy Mode support)
# =============================================================================

def compute_stable_rank(W: torch.Tensor, subsample_limit: Optional[int] = None) -> float:
    """
    Compute the Stable Rank of a weight matrix.
    
    Stable Rank = ||W||_F² / ||W||_2²
    
    Args:
        W: Weight tensor
        subsample_limit: If set, subsample matrices larger than this (Biopsy Mode)
        
    Returns:
        Stable rank as a float >= 1.0
    """
    if W.dim() != 2:
        return float(min(W.shape[-2:]) if W.dim() > 2 else W.numel())
    
    with torch.no_grad():
        try:
            # Biopsy Mode: subsample large matrices
            W_sample = W if subsample_limit is None else _subsample_matrix(W, subsample_limit)
            
            frob_sq = torch.sum(W_sample ** 2).item()
            spectral_sq = torch.linalg.svdvals(W_sample)[0].item() ** 2
            
            if spectral_sq > 1e-10:
                return max(1.0, frob_sq / spectral_sq)
            return 1.0
        except Exception:
            return 1.0


def compute_stable_rank_ratio(W: torch.Tensor, subsample_limit: Optional[int] = None) -> float:
    """
    Compute stable rank as a fraction of maximum possible rank.
    
    Args:
        W: Weight tensor
        subsample_limit: If set, subsample matrices larger than this
        
    Returns:
        Value in [0, 1]
    """
    if W.dim() != 2:
        return 1.0
    
    # Use subsampled dimensions for ratio calculation
    if subsample_limit is not None:
        effective_shape = (min(W.shape[0], subsample_limit), min(W.shape[1], subsample_limit))
    else:
        effective_shape = W.shape
    
    stable_rank = compute_stable_rank(W, subsample_limit)
    max_rank = min(effective_shape[0], effective_shape[1])
    return stable_rank / max_rank if max_rank > 0 else 0.0


def compute_condition_number(W: torch.Tensor, subsample_limit: Optional[int] = None) -> float:
    """
    Compute the Condition Number (σ_max / σ_min).
    
    Args:
        W: Weight tensor
        subsample_limit: If set, subsample matrices larger than this
        
    Returns:
        Condition number (1.0 for well-conditioned, inf for singular)
    """
    if W.dim() != 2:
        return 1.0
    
    with torch.no_grad():
        try:
            # Biopsy Mode: subsample large matrices
            W_sample = W if subsample_limit is None else _subsample_matrix(W, subsample_limit)
            
            s = torch.linalg.svdvals(W_sample)
            s_max = s[0].item()
            s_min = s[-1].item()
            
            if s_min > 1e-10:
                return s_max / s_min
            return float('inf')
        except Exception:
            return float('inf')


def compute_spectral_health(W: torch.Tensor, subsample_limit: Optional[int] = None) -> Dict[str, Any]:
    """Compute comprehensive spectral health metrics."""
    if W.dim() != 2:
        return {
            'stable_rank': 1.0,
            'stable_rank_ratio': 1.0,
            'effective_rank': 1.0,
            'condition': 1.0,
            'spectral_norm': W.norm().item(),
            'min_singular_value': 0.0,
            'frobenius_norm': W.norm().item(),
            'shape': tuple(W.shape),
            'subsampled': False,
        }
    
    # Check if we'll subsample
    subsampled = subsample_limit is not None and (W.shape[0] > subsample_limit or W.shape[1] > subsample_limit)
    
    stable_rank = compute_stable_rank(W, subsample_limit)
    stable_rank_ratio = compute_stable_rank_ratio(W, subsample_limit)
    condition = compute_condition_number(W, subsample_limit)
    
    with torch.no_grad():
        try:
            W_sample = W if subsample_limit is None else _subsample_matrix(W, subsample_limit)
            s = torch.linalg.svdvals(W_sample)
            spectral_norm = s[0].item()
            min_sv = s[-1].item()
            
            s_norm = s / s.sum()
            s_norm = s_norm[s_norm > 1e-10]
            entropy = -torch.sum(s_norm * torch.log(s_norm)).item()
            effective_rank = float(np.exp(entropy))
        except Exception:
            spectral_norm = W.norm().item()
            min_sv = 0.0
            effective_rank = stable_rank
    
    return {
        'stable_rank': stable_rank,
        'stable_rank_ratio': stable_rank_ratio,
        'effective_rank': effective_rank,
        'condition': condition,
        'spectral_norm': spectral_norm,
        'min_singular_value': min_sv,
        'frobenius_norm': W.norm().item(),
        'shape': tuple(W.shape),
        'subsampled': subsampled,
    }


# =============================================================================
# AUTO-CALIBRATION
# =============================================================================

def auto_calibrate(
    model: nn.Module,
    percentile: float = 10.0,
    margin_sr: float = 0.8,
    margin_cond: float = 1.5,
    base_lambda: float = 1e-5,
    sample_ratio: float = 0.05,
    sample_freq: int = 2,
    subsample_limit: int = 1024,
    verbose: bool = True,
) -> ASSRConfig:
    """
    Auto-calibrate ASSR thresholds based on model's spectral distribution.
    
    Uses Biopsy Mode (subsampling) for fast calibration on large models.
    
    Args:
        model: The model to analyze
        percentile: Target the worst X% of layers. Default: 10
        margin_sr: Multiply SR floor by this. Default: 0.8
        margin_cond: Multiply condition ceiling by this. Default: 1.5
        base_lambda: Base regularization strength. Default: 1e-5
        sample_ratio: Fraction of layers to check per step. Default: 0.05
        sample_freq: Check every N steps. Default: 2
        subsample_limit: Max matrix dim for SVD (Biopsy Mode). Default: 1024
        verbose: Print calibration info. Default: True
        
    Returns:
        ASSRConfig with calibrated thresholds
        
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("TinyLlama/...")
        >>> config = auto_calibrate(model)
        >>> trainer = ASSRTrainer(model=model, assr_config=config, ...)
    """
    sr_values = []
    cond_values = []
    n_subsampled = 0
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            W = m.weight
            sr = compute_stable_rank_ratio(W, subsample_limit)
            cond = compute_condition_number(W, subsample_limit)
            sr_values.append(sr)
            if cond < float('inf'):
                cond_values.append(cond)
            
            # Count subsampled layers
            if W.shape[0] > subsample_limit or W.shape[1] > subsample_limit:
                n_subsampled += 1
    
    if not sr_values:
        if verbose:
            print("⚠️ No linear layers found, using defaults")
        return ASSRConfig()
    
    sr_arr = np.array(sr_values)
    cond_arr = np.array(cond_values) if cond_values else np.array([500.0])
    
    # Set thresholds at percentiles with margins
    sr_floor = float(np.percentile(sr_arr, percentile) * margin_sr)
    cond_ceiling = float(np.percentile(cond_arr, 100 - percentile) * margin_cond)
    
    # Clamp to reasonable bounds
    sr_floor = max(0.05, min(sr_floor, 0.4))
    cond_ceiling = max(100, cond_ceiling)
    
    n_sr_trigger = int(np.sum(sr_arr < sr_floor))
    n_cond_trigger = int(np.sum(cond_arr > cond_ceiling))
    
    if verbose:
        print(f"\n🔬 ASSR Auto-Calibration {'(Biopsy Mode)' if n_subsampled > 0 else ''}")
        print(f"   ─────────────────────────────────────────")
        print(f"   Linear layers: {len(sr_values)} ({n_subsampled} subsampled)")
        print(f"   Stable Rank Ratio: [{sr_arr.min():.3f}, {np.median(sr_arr):.3f}, {sr_arr.max():.3f}]")
        print(f"   Condition Number:  [{cond_arr.min():.0f}, {np.median(cond_arr):.0f}, {cond_arr.max():.0f}]")
        print(f"   ─────────────────────────────────────────")
        print(f"   → stable_rank_floor  = {sr_floor:.3f}")
        print(f"   → condition_ceiling  = {cond_ceiling:.0f}")
        print(f"   → subsample_limit    = {subsample_limit}")
        print(f"   Expected init triggers: {n_sr_trigger} rank, {n_cond_trigger} condition")
        print(f"   ─────────────────────────────────────────\n")
    
    return ASSRConfig(
        base_lambda=base_lambda,
        stable_rank_floor=sr_floor,
        condition_ceiling=cond_ceiling,
        sample_ratio=sample_ratio,
        sample_freq=sample_freq,
        max_severity_multiplier=10.0,
        subsample_limit=subsample_limit,
        log_interventions=False,
    )


# =============================================================================
# ASSR TRAINER
# =============================================================================

try:
    from transformers import Trainer
    
    class ASSRTrainer(Trainer):
        """
        Hugging Face Trainer with Auto-Calibrated Stochastic Spectral Regularization.
        
        v1.4 Features:
        - Biopsy Mode: Subsample large matrices for 10x faster spectral computation
        - Auto-calibration: Use auto_calibrate() for optimal thresholds
        
        Example:
            ```python
            from assr import ASSRTrainer, auto_calibrate
            
            config = auto_calibrate(model)
            trainer = ASSRTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                assr_config=config,
            )
            trainer.train()
            
            print(trainer.assr_stats)  # Quick stats access
            ```
        """
        
        def __init__(
            self,
            assr_config: Optional[ASSRConfig] = None,
            *args,
            **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.assr_config = assr_config or ASSRConfig()
            self._linear_layers: Optional[List[nn.Module]] = None
            
            # Statistics
            self._assr_rank_interventions: int = 0
            self._assr_condition_interventions: int = 0
            self._assr_total_reg_loss: float = 0.0
            self._assr_steps_with_intervention: int = 0
        
        @property
        def linear_layers(self) -> List[nn.Module]:
            """Lazily cache linear layers."""
            if self._linear_layers is None:
                self._linear_layers = [
                    m for m in self.model.modules()
                    if isinstance(m, nn.Linear)
                ]
            return self._linear_layers
        
        @property
        def assr_stats(self) -> Dict[str, Any]:
            """Quick access to ASSR statistics (alias for get_assr_summary)."""
            return {
                'rank_int': self._assr_rank_interventions,
                'cond_int': self._assr_condition_interventions,
                'total_int': self._assr_rank_interventions + self._assr_condition_interventions,
                'total_reg_loss': self._assr_total_reg_loss,
                'steps_with_int': self._assr_steps_with_intervention,
            }
        
        def compute_loss(
            self,
            model: nn.Module,
            inputs: Dict[str, torch.Tensor],
            return_outputs: bool = False,
            **kwargs
        ):
            """Compute training loss with ASSR regularization."""
            outputs = model(**inputs)
            main_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            step = self.state.global_step if self.state else 0
            if step % self.assr_config.sample_freq != 0:
                return (main_loss, outputs) if return_outputs else main_loss
            
            reg_loss = self._compute_assr_regularization(
                main_loss.device,
                main_loss.dtype
            )
            
            total_loss = main_loss + reg_loss
            return (total_loss, outputs) if return_outputs else total_loss
        
        def _compute_assr_regularization(
            self,
            device: torch.device,
            dtype: torch.dtype
        ) -> torch.Tensor:
            """Compute ASSR regularization with Biopsy Mode support."""
            cfg = self.assr_config
            reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
            
            layers = self.linear_layers
            num_sample = max(1, int(len(layers) * cfg.sample_ratio))
            subset = random.sample(layers, min(num_sample, len(layers)))
            
            step_had_intervention = False
            
            for m in subset:
                W = m.weight
                
                # Use Biopsy Mode if configured
                limit = cfg.subsample_limit
                
                sr_ratio = compute_stable_rank_ratio(W, limit)
                is_rank_low = sr_ratio < cfg.stable_rank_floor
                
                condition = compute_condition_number(W, limit)
                is_cond_high = condition > cfg.condition_ceiling
                
                if is_rank_low or is_cond_high:
                    step_had_intervention = True
                    
                    if is_rank_low:
                        severity = (cfg.stable_rank_floor - sr_ratio) / cfg.stable_rank_floor
                        severity = min(max(severity, 0.0), 1.0)
                        self._assr_rank_interventions += 1
                        trigger_type = "Rank"
                    else:
                        excess = condition / cfg.condition_ceiling
                        severity = min((excess - 1.0) / 10.0, 1.0)
                        self._assr_condition_interventions += 1
                        trigger_type = "Cond"
                    
                    adaptive_lambda = cfg.base_lambda * (
                        1.0 + cfg.max_severity_multiplier * severity
                    )
                    
                    # L2 penalty on FULL matrix (not subsampled)
                    penalty = torch.norm(W) ** 2
                    reg_loss = reg_loss + adaptive_lambda * penalty
                    
                    if cfg.log_interventions:
                        print(f"  [ASSR] {trigger_type}: SR={sr_ratio:.3f}, "
                              f"C={condition:.1f}, λ={adaptive_lambda:.2e}")
            
            if step_had_intervention:
                self._assr_steps_with_intervention += 1
            
            self._assr_total_reg_loss += reg_loss.item()
            return reg_loss
        
        def get_assr_summary(self) -> Dict[str, Any]:
            """Get detailed ASSR statistics."""
            total_steps = self.state.global_step if self.state else 0
            return {
                'rank_interventions': self._assr_rank_interventions,
                'condition_interventions': self._assr_condition_interventions,
                'total_interventions': (
                    self._assr_rank_interventions + self._assr_condition_interventions
                ),
                'steps_with_intervention': self._assr_steps_with_intervention,
                'intervention_rate': (
                    self._assr_steps_with_intervention / max(total_steps, 1)
                ),
                'total_reg_loss': self._assr_total_reg_loss,
                'avg_reg_loss_per_step': (
                    self._assr_total_reg_loss / max(total_steps, 1)
                ),
                'num_linear_layers': len(self.linear_layers),
                'config': self.assr_config,
            }
        
        def reset_assr_stats(self) -> None:
            """Reset ASSR statistics."""
            self._assr_rank_interventions = 0
            self._assr_condition_interventions = 0
            self._assr_total_reg_loss = 0.0
            self._assr_steps_with_intervention = 0

except ImportError:
    ASSRTrainer = None


# =============================================================================
# UTILITIES
# =============================================================================

def print_spectral_report(
    model: nn.Module, 
    top_k: int = 10,
    subsample_limit: Optional[int] = 1024
) -> None:
    """
    Print spectral health report for all linear layers.
    
    Args:
        model: PyTorch model to analyze
        top_k: Number of worst layers to show
        subsample_limit: Max matrix dim for SVD (Biopsy Mode). Default: 1024
    """
    results = []
    n_subsampled = 0
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            health = compute_spectral_health(m.weight, subsample_limit)
            results.append({'name': name, **health})
            if health['subsampled']:
                n_subsampled += 1
    
    print("\n" + "=" * 75)
    print(f"  SPECTRAL HEALTH REPORT {'(Biopsy Mode)' if n_subsampled > 0 else ''}")
    print("=" * 75)
    print(f"  Total Linear Layers: {len(results)} ({n_subsampled} subsampled)")
    
    if not results:
        print("  No linear layers found.")
        print("=" * 75 + "\n")
        return
    
    sr_vals = [r['stable_rank_ratio'] for r in results]
    c_vals = [r['condition'] for r in results if r['condition'] < float('inf')]
    
    print(f"  Stable Rank Ratio: min={min(sr_vals):.3f}, max={max(sr_vals):.3f}, "
          f"mean={sum(sr_vals)/len(sr_vals):.3f}")
    if c_vals:
        print(f"  Condition Number:  min={min(c_vals):.1f}, max={max(c_vals):.1f}, "
              f"mean={sum(c_vals)/len(c_vals):.1f}")
    
    results.sort(key=lambda x: x['stable_rank_ratio'])
    
    print(f"\n  {'Layer':<40} {'Shape':<15} {'SR Ratio':<10} {'Condition':<12}")
    print("  " + "-" * 73)
    
    for r in results[:top_k]:
        sr_flag = "⚠️" if r['stable_rank_ratio'] < 0.25 else "  "
        c_flag = "⚠️" if r['condition'] > 500 else "  "
        
        name = r['name'][-38:] if len(r['name']) > 38 else r['name']
        shape_str = f"{r['shape']}"
        cond_str = f"{r['condition']:.1f}" if r['condition'] < 1e6 else "inf"
        
        print(f"  {sr_flag}{c_flag}{name:<38} {shape_str:<15} "
              f"{r['stable_rank_ratio']:<10.3f} {cond_str:<12}")
    
    if len(results) > top_k:
        print(f"  ... and {len(results) - top_k} more layers")
    
    n_rank_issues = sum(1 for r in results if r['stable_rank_ratio'] < 0.25)
    n_cond_issues = sum(1 for r in results if r['condition'] > 500)
    
    print("\n  " + "-" * 73)
    if n_rank_issues == 0 and n_cond_issues == 0:
        print("  ✅ All layers appear healthy")
    else:
        if n_rank_issues > 0:
            print(f"  ⚠️  {n_rank_issues} layer(s) have low stable rank")
        if n_cond_issues > 0:
            print(f"  ⚠️  {n_cond_issues} layer(s) are ill-conditioned")
        print("  💡 Tip: Use auto_calibrate(model) for large models")
    
    print("=" * 75 + "\n")


def analyze_layer(
    layer: nn.Linear, 
    name: str = "layer",
    subsample_limit: Optional[int] = 1024
) -> Dict[str, Any]:
    """Detailed spectral analysis of a single layer."""
    health = compute_spectral_health(layer.weight, subsample_limit)
    
    print(f"\n  Analysis of '{name}':")
    print(f"    Shape: {health['shape']} {'(subsampled)' if health['subsampled'] else ''}")
    print(f"    Stable Rank: {health['stable_rank']:.2f} (ratio: {health['stable_rank_ratio']:.3f})")
    print(f"    Effective Rank: {health['effective_rank']:.2f}")
    print(f"    Condition Number: {health['condition']:.1f}")
    print(f"    Spectral Norm: {health['spectral_norm']:.4f}")
    print(f"    Frobenius Norm: {health['frobenius_norm']:.4f}")
    
    issues = []
    if health['stable_rank_ratio'] < 0.25:
        issues.append("Low stable rank")
    if health['condition'] > 500:
        issues.append("High condition number")
    
    if issues:
        print(f"    ⚠️  Issues: {', '.join(issues)}")
    else:
        print(f"    ✅ Layer appears healthy")
    
    return health
