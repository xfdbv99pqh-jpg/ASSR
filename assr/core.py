# =============================================================================
# ASSR: Auto-Calibrated Stochastic Spectral Regularization
# Version: 1.5.0
# =============================================================================
#
# v1.5: True auto-configuration - detects model size and adjusts ALL parameters:
#   - Thresholds (stable_rank_floor, condition_ceiling)
#   - Speed (sample_ratio, sample_freq, subsample_limit)
#
# =============================================================================

import torch
import torch.nn as nn
import random
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

__version__ = "1.5.0"
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
    
    For best results, use auto_calibrate(model) which sets ALL parameters
    based on model size automatically.
    """
    
    base_lambda: float = 1e-4
    stable_rank_floor: float = 0.25
    condition_ceiling: float = 500.0
    sample_ratio: float = 0.1
    sample_freq: int = 1
    max_severity_multiplier: float = 10.0
    subsample_limit: Optional[int] = 1024
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
    """Subsample large matrix for faster SVD."""
    m, n = W.shape
    
    if m <= limit and n <= limit:
        return W
    
    with torch.no_grad():
        if m > limit:
            row_idx = torch.randperm(m, device=W.device)[:limit]
            W = W[row_idx, :]
        if n > limit:
            col_idx = torch.randperm(n, device=W.device)[:limit]
            W = W[:, col_idx]
    
    return W


# =============================================================================
# SPECTRAL SENSORS
# =============================================================================

def compute_stable_rank(W: torch.Tensor, subsample_limit: Optional[int] = None) -> float:
    """Compute Stable Rank = ||W||_F² / ||W||_2²"""
    if W.dim() != 2:
        return float(min(W.shape[-2:]) if W.dim() > 2 else W.numel())
    
    with torch.no_grad():
        try:
            W_sample = W if subsample_limit is None else _subsample_matrix(W, subsample_limit)
            frob_sq = torch.sum(W_sample ** 2).item()
            spectral_sq = torch.linalg.svdvals(W_sample)[0].item() ** 2
            if spectral_sq > 1e-10:
                return max(1.0, frob_sq / spectral_sq)
            return 1.0
        except Exception:
            return 1.0


def compute_stable_rank_ratio(W: torch.Tensor, subsample_limit: Optional[int] = None) -> float:
    """Compute stable rank as fraction of max possible rank [0, 1]."""
    if W.dim() != 2:
        return 1.0
    
    if subsample_limit is not None:
        effective_shape = (min(W.shape[0], subsample_limit), min(W.shape[1], subsample_limit))
    else:
        effective_shape = W.shape
    
    stable_rank = compute_stable_rank(W, subsample_limit)
    max_rank = min(effective_shape[0], effective_shape[1])
    return stable_rank / max_rank if max_rank > 0 else 0.0


def compute_condition_number(W: torch.Tensor, subsample_limit: Optional[int] = None) -> float:
    """Compute Condition Number = σ_max / σ_min."""
    if W.dim() != 2:
        return 1.0
    
    with torch.no_grad():
        try:
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
            'stable_rank': 1.0, 'stable_rank_ratio': 1.0, 'effective_rank': 1.0,
            'condition': 1.0, 'spectral_norm': W.norm().item(),
            'min_singular_value': 0.0, 'frobenius_norm': W.norm().item(),
            'shape': tuple(W.shape), 'subsampled': False,
        }
    
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
        'stable_rank': stable_rank, 'stable_rank_ratio': stable_rank_ratio,
        'effective_rank': effective_rank, 'condition': condition,
        'spectral_norm': spectral_norm, 'min_singular_value': min_sv,
        'frobenius_norm': W.norm().item(), 'shape': tuple(W.shape),
        'subsampled': subsampled,
    }


# =============================================================================
# AUTO-CALIBRATION (Now truly auto-configures EVERYTHING)
# =============================================================================

def auto_calibrate(
    model: nn.Module,
    percentile: float = 10.0,
    verbose: bool = True,
) -> ASSRConfig:
    """
    Auto-calibrate ALL ASSR parameters based on model size and spectral distribution.
    
    This function automatically determines:
    1. Thresholds (stable_rank_floor, condition_ceiling) - based on spectral stats
    2. Speed parameters (sample_ratio, sample_freq, subsample_limit) - based on model size
    
    Args:
        model: The model to analyze
        percentile: Target the worst X% of layers for thresholds. Default: 10
        verbose: Print calibration info. Default: True
        
    Returns:
        Fully configured ASSRConfig
        
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("TinyLlama/...")
        >>> config = auto_calibrate(model)  # That's it! Everything is configured.
        >>> trainer = ASSRTrainer(model=model, assr_config=config, ...)
    """
    
    # =========================================================================
    # STEP 1: Analyze model size
    # =========================================================================
    
    linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    num_layers = len(linear_layers)
    
    if num_layers == 0:
        if verbose:
            print("⚠️ No linear layers found, using defaults")
        return ASSRConfig()
    
    # Count parameters and find max layer size
    total_params = sum(p.numel() for p in model.parameters())
    max_dim = max(max(m.weight.shape) for _, m in linear_layers)
    
    # =========================================================================
    # STEP 2: Determine speed parameters based on model size
    # =========================================================================
    
    # Model size tiers:
    # - Small (<100M): Full speed, check frequently
    # - Medium (100M-500M): Moderate sampling
    # - Large (500M-2B): Light sampling, less frequent
    # - XL (>2B): Minimal sampling, very infrequent
    
    if total_params < 100_000_000:  # <100M
        sample_ratio = 0.10   # 10% of layers
        sample_freq = 1       # Every step
        subsample_limit = 1024
        base_lambda = 1e-4
        tier = "Small (<100M)"
        
    elif total_params < 500_000_000:  # 100M-500M
        sample_ratio = 0.05   # 5% of layers
        sample_freq = 2       # Every 2 steps
        subsample_limit = 768
        base_lambda = 5e-5
        tier = "Medium (100M-500M)"
        
    elif total_params < 2_000_000_000:  # 500M-2B
        sample_ratio = 0.02   # 2% of layers
        sample_freq = 5       # Every 5 steps
        subsample_limit = 512
        base_lambda = 2e-5
        tier = "Large (500M-2B)"
        
    else:  # >2B
        sample_ratio = 0.01   # 1% of layers
        sample_freq = 10      # Every 10 steps
        subsample_limit = 384
        base_lambda = 1e-5
        tier = "XL (>2B)"
    
    # Ensure we check at least 1 layer
    layers_per_check = max(1, int(num_layers * sample_ratio))
    
    # =========================================================================
    # STEP 3: Collect spectral statistics (using biopsy mode for speed)
    # =========================================================================
    
    sr_values = []
    cond_values = []
    n_subsampled = 0
    
    # Only sample subset for large models during calibration too
    if num_layers > 50:
        calibration_layers = random.sample(linear_layers, min(50, num_layers))
    else:
        calibration_layers = linear_layers
    
    for name, m in calibration_layers:
        W = m.weight
        sr = compute_stable_rank_ratio(W, subsample_limit)
        cond = compute_condition_number(W, subsample_limit)
        sr_values.append(sr)
        if cond < float('inf'):
            cond_values.append(cond)
        if W.shape[0] > subsample_limit or W.shape[1] > subsample_limit:
            n_subsampled += 1
    
    sr_arr = np.array(sr_values)
    cond_arr = np.array(cond_values) if cond_values else np.array([500.0])
    
    # =========================================================================
    # STEP 4: Set thresholds at percentiles with margins
    # =========================================================================
    
    sr_floor = float(np.percentile(sr_arr, percentile) * 0.8)
    cond_ceiling = float(np.percentile(cond_arr, 100 - percentile) * 1.5)
    
    # Clamp to reasonable bounds
    sr_floor = max(0.05, min(sr_floor, 0.4))
    cond_ceiling = max(100, cond_ceiling)
    
    # Expected triggers at init
    n_sr_trigger = int(np.sum(sr_arr < sr_floor))
    n_cond_trigger = int(np.sum(cond_arr > cond_ceiling))
    
    # =========================================================================
    # STEP 5: Report
    # =========================================================================
    
    if verbose:
        print(f"\n🔬 ASSR Auto-Calibration v1.5")
        print(f"   ─────────────────────────────────────────────────")
        print(f"   Model tier: {tier}")
        print(f"   Parameters: {total_params/1e6:.1f}M")
        print(f"   Linear layers: {num_layers} (max dim: {max_dim})")
        print(f"   ─────────────────────────────────────────────────")
        print(f"   Speed Settings (auto-tuned for model size):")
        print(f"   → sample_ratio     = {sample_ratio} ({layers_per_check} layers/check)")
        print(f"   → sample_freq      = {sample_freq} (every {sample_freq} steps)")
        print(f"   → subsample_limit  = {subsample_limit} (biopsy size)")
        print(f"   → base_lambda      = {base_lambda}")
        print(f"   ─────────────────────────────────────────────────")
        print(f"   Threshold Settings (from spectral distribution):")
        print(f"   → stable_rank_floor  = {sr_floor:.3f}")
        print(f"   → condition_ceiling  = {cond_ceiling:.0f}")
        print(f"   ─────────────────────────────────────────────────")
        print(f"   Spectral stats (sampled {len(calibration_layers)} layers):")
        print(f"   SR ratio: [{sr_arr.min():.3f}, {np.median(sr_arr):.3f}, {sr_arr.max():.3f}]")
        print(f"   Condition: [{cond_arr.min():.0f}, {np.median(cond_arr):.0f}, {cond_arr.max():.0f}]")
        print(f"   Expected init triggers: {n_sr_trigger} rank, {n_cond_trigger} cond")
        print(f"   ─────────────────────────────────────────────────\n")
    
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
        HuggingFace Trainer with Auto-Calibrated Stochastic Spectral Regularization.
        
        Usage:
            config = auto_calibrate(model)  # Auto-configures everything
            trainer = ASSRTrainer(model=model, args=args, assr_config=config, ...)
            trainer.train()
            print(trainer.assr_stats)
        """
        
        def __init__(self, assr_config: Optional[ASSRConfig] = None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.assr_config = assr_config or ASSRConfig()
            self._linear_layers: Optional[List[nn.Module]] = None
            self._assr_rank_interventions: int = 0
            self._assr_condition_interventions: int = 0
            self._assr_total_reg_loss: float = 0.0
            self._assr_steps_with_intervention: int = 0
        
        @property
        def linear_layers(self) -> List[nn.Module]:
            if self._linear_layers is None:
                self._linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
            return self._linear_layers
        
        @property
        def assr_stats(self) -> Dict[str, Any]:
            """Quick stats access."""
            return {
                'rank_int': self._assr_rank_interventions,
                'cond_int': self._assr_condition_interventions,
                'total_int': self._assr_rank_interventions + self._assr_condition_interventions,
                'total_reg_loss': self._assr_total_reg_loss,
                'steps_with_int': self._assr_steps_with_intervention,
            }
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            main_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            step = self.state.global_step if self.state else 0
            if step % self.assr_config.sample_freq != 0:
                return (main_loss, outputs) if return_outputs else main_loss
            
            reg_loss = self._compute_assr_regularization(main_loss.device, main_loss.dtype)
            total_loss = main_loss + reg_loss
            return (total_loss, outputs) if return_outputs else total_loss
        
        def _compute_assr_regularization(self, device, dtype) -> torch.Tensor:
            cfg = self.assr_config
            reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
            
            layers = self.linear_layers
            num_sample = max(1, int(len(layers) * cfg.sample_ratio))
            subset = random.sample(layers, min(num_sample, len(layers)))
            
            step_had_intervention = False
            
            for m in subset:
                W = m.weight
                limit = cfg.subsample_limit
                
                sr_ratio = compute_stable_rank_ratio(W, limit)
                is_rank_low = sr_ratio < cfg.stable_rank_floor
                
                condition = compute_condition_number(W, limit)
                is_cond_high = condition > cfg.condition_ceiling
                
                if is_rank_low or is_cond_high:
                    step_had_intervention = True
                    
                    if is_rank_low:
                        severity = min(max((cfg.stable_rank_floor - sr_ratio) / cfg.stable_rank_floor, 0.0), 1.0)
                        self._assr_rank_interventions += 1
                    else:
                        severity = min((condition / cfg.condition_ceiling - 1.0) / 10.0, 1.0)
                        self._assr_condition_interventions += 1
                    
                    adaptive_lambda = cfg.base_lambda * (1.0 + cfg.max_severity_multiplier * severity)
                    penalty = torch.norm(W) ** 2
                    reg_loss = reg_loss + adaptive_lambda * penalty
                    
                    if cfg.log_interventions:
                        print(f"  [ASSR] SR={sr_ratio:.3f}, C={condition:.1f}, λ={adaptive_lambda:.2e}")
            
            if step_had_intervention:
                self._assr_steps_with_intervention += 1
            
            self._assr_total_reg_loss += reg_loss.item()
            return reg_loss
        
        def get_assr_summary(self) -> Dict[str, Any]:
            total_steps = self.state.global_step if self.state else 0
            return {
                'rank_interventions': self._assr_rank_interventions,
                'condition_interventions': self._assr_condition_interventions,
                'total_interventions': self._assr_rank_interventions + self._assr_condition_interventions,
                'steps_with_intervention': self._assr_steps_with_intervention,
                'intervention_rate': self._assr_steps_with_intervention / max(total_steps, 1),
                'total_reg_loss': self._assr_total_reg_loss,
                'avg_reg_loss_per_step': self._assr_total_reg_loss / max(total_steps, 1),
                'num_linear_layers': len(self.linear_layers),
                'config': self.assr_config,
            }
        
        def reset_assr_stats(self) -> None:
            self._assr_rank_interventions = 0
            self._assr_condition_interventions = 0
            self._assr_total_reg_loss = 0.0
            self._assr_steps_with_intervention = 0

except ImportError:
    ASSRTrainer = None


# =============================================================================
# UTILITIES
# =============================================================================

def print_spectral_report(model: nn.Module, top_k: int = 10, subsample_limit: Optional[int] = 1024) -> None:
    """Print spectral health report."""
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
    print(f"  Linear Layers: {len(results)} ({n_subsampled} subsampled)")
    
    if not results:
        print("  No linear layers found.")
        print("=" * 75 + "\n")
        return
    
    sr_vals = [r['stable_rank_ratio'] for r in results]
    c_vals = [r['condition'] for r in results if r['condition'] < float('inf')]
    
    print(f"  SR Ratio: [{min(sr_vals):.3f}, {np.median(sr_vals):.3f}, {max(sr_vals):.3f}]")
    if c_vals:
        print(f"  Condition: [{min(c_vals):.0f}, {np.median(c_vals):.0f}, {max(c_vals):.0f}]")
    
    results.sort(key=lambda x: x['stable_rank_ratio'])
    
    print(f"\n  {'Layer':<40} {'Shape':<15} {'SR':>8} {'Cond':>10}")
    print("  " + "-" * 73)
    
    for r in results[:top_k]:
        sr_flag = "⚠️" if r['stable_rank_ratio'] < 0.25 else "  "
        c_flag = "⚠️" if r['condition'] > 500 else "  "
        name = r['name'][-38:] if len(r['name']) > 38 else r['name']
        cond_str = f"{r['condition']:.0f}" if r['condition'] < 1e6 else "inf"
        print(f"  {sr_flag}{c_flag}{name:<38} {str(r['shape']):<15} {r['stable_rank_ratio']:>8.3f} {cond_str:>10}")
    
    if len(results) > top_k:
        print(f"  ... and {len(results) - top_k} more layers")
    
    print("=" * 75 + "\n")


def analyze_layer(layer: nn.Linear, name: str = "layer", subsample_limit: Optional[int] = 1024) -> Dict[str, Any]:
    """Detailed analysis of a single layer."""
    health = compute_spectral_health(layer.weight, subsample_limit)
    print(f"\n  '{name}': shape={health['shape']}, SR={health['stable_rank_ratio']:.3f}, Cond={health['condition']:.0f}")
    return health
