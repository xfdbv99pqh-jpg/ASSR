# =============================================================================
# ASSR: Auto-Calibrated Stochastic Spectral Regularization
# Version: 1.3.1 (A100/BFloat16 Hotfix)
# =============================================================================

import torch
import torch.nn as nn
import random
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

__version__ = "1.3.1"
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
    """
    base_lambda: float = 1e-4
    stable_rank_floor: float = 0.25
    condition_ceiling: float = 500.0
    sample_ratio: float = 0.1
    sample_freq: int = 1
    max_severity_multiplier: float = 10.0
    log_interventions: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert 0 < self.sample_ratio <= 1.0, "sample_ratio must be in (0, 1]"
        assert self.base_lambda >= 0, "base_lambda must be non-negative"
        assert 0 < self.stable_rank_floor < 1, "stable_rank_floor must be in (0, 1)"
        assert self.condition_ceiling > 1, "condition_ceiling must be > 1"
        assert self.sample_freq >= 1, "sample_freq must be >= 1"
        assert self.max_severity_multiplier >= 0, "max_severity_multiplier must be >= 0"


# =============================================================================
# SPECTRAL SENSORS (PATCHED FOR A100/BF16)
# =============================================================================

def compute_stable_rank(W: torch.Tensor) -> float:
    """
    Compute the Stable Rank of a weight matrix.
    Stable Rank = ||W||_F^2 / ||W||_2^2
    """
    if W.dim() != 2:
        return float(min(W.shape[-2:]) if W.dim() > 2 else W.numel())
    
    # CRITICAL FIX: Cast to float32 for stable SVD on Ampere/Hopper GPUs
    W_f = W.float()
    
    with torch.no_grad():
        try:
            frob_sq = torch.sum(W_f ** 2).item()
            # SVD is significantly more stable in FP32
            s = torch.linalg.svdvals(W_f)
            spectral_sq = s[0].item() ** 2
            
            if spectral_sq > 1e-10:
                return max(1.0, frob_sq / spectral_sq)
            return 1.0
        except Exception:
            return 1.0


def compute_stable_rank_ratio(W: torch.Tensor) -> float:
    """
    Compute stable rank as a fraction of maximum possible rank.
    """
    if W.dim() != 2: return 1.0
    
    # Internal function already handles the float cast
    stable_rank = compute_stable_rank(W) 
    max_rank = min(W.shape[0], W.shape[1])
    return stable_rank / max_rank if max_rank > 0 else 0.0


def compute_condition_number(W: torch.Tensor) -> float:
    """
    Compute the Condition Number (sigma_max / sigma_min).
    """
    if W.dim() != 2: return 1.0
    
    # CRITICAL FIX: Cast to float32
    W_f = W.float()
    
    with torch.no_grad():
        try:
            s = torch.linalg.svdvals(W_f)
            s_max = s[0].item()
            s_min = s[-1].item()
            if s_min > 1e-10:
                return s_max / s_min
            return float('inf')
        except Exception:
            return float('inf')


def compute_spectral_health(W: torch.Tensor) -> Dict[str, Any]:
    """Compute comprehensive spectral health metrics."""
    if W.dim() != 2:
        return {'stable_rank': 1.0, 'shape': tuple(W.shape)}
    
    # Use the patched functions for consistency
    stable_rank = compute_stable_rank(W)
    stable_rank_ratio = compute_stable_rank_ratio(W)
    condition = compute_condition_number(W)
    
    # CRITICAL FIX: Cast to float32 for detailed stats
    W_f = W.float()
    
    with torch.no_grad():
        try:
            s = torch.linalg.svdvals(W_f)
            spectral_norm = s[0].item()
            min_sv = s[-1].item()
            
            s_norm = s / s.sum()
            s_norm = s_norm[s_norm > 1e-10]
            entropy = -torch.sum(s_norm * torch.log(s_norm)).item()
            effective_rank = float(np.exp(entropy))
        except Exception:
            spectral_norm = W_f.norm().item()
            min_sv = 0.0
            effective_rank = stable_rank
    
    return {
        'stable_rank': stable_rank,
        'stable_rank_ratio': stable_rank_ratio,
        'effective_rank': effective_rank,
        'condition': condition,
        'spectral_norm': spectral_norm,
        'min_singular_value': min_sv,
        'frobenius_norm': W_f.norm().item(),
        'shape': tuple(W.shape),
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
    verbose: bool = True,
) -> ASSRConfig:
    """
    Auto-calibrate ASSR thresholds based on model's spectral distribution.
    """
    sr_values = []
    cond_values = []
    
    # Collect spectral metrics from all linear layers
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            # Using the fixed/patched functions
            sr = compute_stable_rank_ratio(m.weight)
            # Filter out dead inputs (0.0) or single-dim fallbacks (approx 0.0)
            if sr > 0.001: 
                sr_values.append(sr)
            
            cond = compute_condition_number(m.weight)
            if cond < float('inf'):
                cond_values.append(cond)
    
    if not sr_values:
        if verbose:
            print("⚠️ No linear layers found or model weights are zero, using defaults")
        return ASSRConfig()
    
    sr_arr = np.array(sr_values)
    cond_arr = np.array(cond_values) if cond_values else np.array([500.0])
    
    # Set thresholds at percentiles with margins
    sr_floor = float(np.percentile(sr_arr, percentile) * margin_sr)
    cond_ceiling = float(np.percentile(cond_arr, 100 - percentile) * margin_cond)
    
    # Ensure reasonable bounds
    sr_floor = max(0.05, min(sr_floor, 0.4))  # Clamp to [0.05, 0.4]
    cond_ceiling = max(100, cond_ceiling)       # At least 100
    
    if verbose:
        print(f"\n🔬 ASSR Auto-Calibration (FP32 Precision)")
        print(f"   ─────────────────────────────────────────")
        print(f"   Linear layers analyzed: {len(sr_values)}")
        print(f"   Stable Rank Ratio: min={sr_arr.min():.3f}, "
              f"median={np.median(sr_arr):.3f}, max={sr_arr.max():.3f}")
        print(f"   Condition Number:  min={cond_arr.min():.0f}, "
              f"median={np.median(cond_arr):.0f}, max={cond_arr.max():.0f}")
        print(f"   ─────────────────────────────────────────")
        print(f"   Calibrated thresholds (targeting {percentile}% outliers):")
        print(f"   → stable_rank_floor  = {sr_floor:.3f}")
        print(f"   → condition_ceiling  = {cond_ceiling:.0f}")
        print(f"   ─────────────────────────────────────────\n")
    
    return ASSRConfig(
        base_lambda=base_lambda,
        stable_rank_floor=sr_floor,
        condition_ceiling=cond_ceiling,
        sample_ratio=sample_ratio,
        sample_freq=sample_freq,
        max_severity_multiplier=10.0,
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
            self.assr_rank_interventions: int = 0
            self.assr_condition_interventions: int = 0
            self.assr_total_reg_loss: float = 0.0
            self.assr_steps_with_intervention: int = 0
        
        @property
        def linear_layers(self) -> List[nn.Module]:
            """Lazily cache linear layers from the model."""
            if self._linear_layers is None:
                self._linear_layers = [
                    m for m in self.model.modules()
                    if isinstance(m, nn.Linear)
                ]
            return self._linear_layers
        
        def compute_loss(
            self,
            model: nn.Module,
            inputs: Dict[str, torch.Tensor],
            return_outputs: bool = False,
            **kwargs
        ):
            """Compute training loss with ASSR regularization."""
            # Forward pass
            outputs = model(**inputs)
            main_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # Check if we should run ASSR this step
            step = self.state.global_step if self.state else 0
            if step % self.assr_config.sample_freq != 0:
                return (main_loss, outputs) if return_outputs else main_loss
            
            # Compute ASSR regularization
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
            """Compute the ASSR regularization term."""
            cfg = self.assr_config
            reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
            
            # Stochastic sampling of layers
            layers = self.linear_layers
            num_sample = max(1, int(len(layers) * cfg.sample_ratio))
            subset = random.sample(layers, min(num_sample, len(layers)))
            
            step_had_intervention = False
            
            for m in subset:
                W = m.weight
                
                # --- SENSOR 1: Stable Rank (Collapse Detection) ---
                sr_ratio = compute_stable_rank_ratio(W)
                is_rank_low = sr_ratio < cfg.stable_rank_floor
                
                # --- SENSOR 2: Condition Number (Ill-conditioning) ---
                condition = compute_condition_number(W)
                is_cond_high = condition > cfg.condition_ceiling
                
                # Apply regularization if either sensor triggers
                if is_rank_low or is_cond_high:
                    step_had_intervention = True
                    
                    # Compute severity in [0, 1]
                    if is_rank_low:
                        severity = (cfg.stable_rank_floor - sr_ratio) / cfg.stable_rank_floor
                        self.assr_rank_interventions += 1
                        trigger_type = "Rank"
                    else:
                        excess = condition / cfg.condition_ceiling
                        severity = min((excess - 1.0) / 10.0, 1.0)
                        self.assr_condition_interventions += 1
                        trigger_type = "Cond"
                    
                    # Clamp severity
                    severity = max(0.0, min(severity, 1.0))

                    # Adaptive lambda: scales with severity
                    adaptive_lambda = cfg.base_lambda * (
                        1.0 + cfg.max_severity_multiplier * severity
                    )
                    
                    # L2 penalty (Frobenius norm squared)
                    penalty = torch.norm(W) ** 2
                    reg_loss = reg_loss + adaptive_lambda * penalty
                    
                    # Debug logging
                    if cfg.log_interventions:
                        print(
                            f"  [ASSR] {trigger_type}: "
                            f"SR={sr_ratio:.3f}, C={condition:.1f}, "
                            f"sev={severity:.2f}, λ={adaptive_lambda:.2e}"
                        )
            
            if step_had_intervention:
                self.assr_steps_with_intervention += 1
            
            self.assr_total_reg_loss += reg_loss.item()
            return reg_loss
        
        def get_assr_summary(self) -> Dict[str, Any]:
            """Get summary statistics of ASSR activity during training."""
            total_steps = self.state.global_step if self.state else 0
            return {
                'rank_interventions': self.assr_rank_interventions,
                'condition_interventions': self.assr_condition_interventions,
                'total_reg_loss': self.assr_total_reg_loss,
            }
        
        def reset_assr_stats(self) -> None:
            """Reset ASSR statistics counters."""
            self.assr_rank_interventions = 0
            self.assr_condition_interventions = 0
            self.assr_total_reg_loss = 0.0
            self.assr_steps_with_intervention = 0

except ImportError:
    ASSRTrainer = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_spectral_report(model: nn.Module, top_k: int = 10) -> None:
    """Print a spectral health report for all linear layers."""
    results = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            health = compute_spectral_health(m.weight)
            results.append({'name': name, **health})
    
    print("\n" + "=" * 75)
    print("  SPECTRAL HEALTH REPORT (FP32 Precision)")
    print("=" * 75)
    print(f"  Total Linear Layers: {len(results)}")
    
    if not results:
        print("  No linear layers found.")
        return
    
    sr_vals = [r['stable_rank_ratio'] for r in results]
    print(f"  Stable Rank Ratio: min={min(sr_vals):.3f}, max={max(sr_vals):.3f}")
    
    results.sort(key=lambda x: x['stable_rank_ratio'])
    
    print(f"\n  {'Layer':<40} {'Shape':<15} {'SR Ratio':<10} {'Condition':<12}")
    print("  " + "-" * 73)
    
    for r in results[:top_k]:
        sr_flag = "⚠️" if r['stable_rank_ratio'] < 0.25 else "  "
        c_flag = "⚠️" if r['condition'] > 500 else "  "
        name = r['name']
        if len(name) > 38: name = "..." + name[-35:]
        cond_str = f"{r['condition']:.1f}" if r['condition'] < 1e6 else "inf"
        
        print(f"  {sr_flag}{c_flag}{name:<38} {f'{r['shape']}':<15} "
              f"{r['stable_rank_ratio']:<10.3f} {cond_str:<12}")
    
    print("=" * 75 + "\n")

def analyze_layer(layer: nn.Linear, name: str = "layer") -> Dict[str, Any]:
    health = compute_spectral_health(layer.weight)
    print(f"\n  Analysis of '{name}':")
    print(f"    Stable Rank Ratio: {health['stable_rank_ratio']:.3f}")
    return health
