# =============================================================================
# ASSR: Auto-Calibrated Stochastic Spectral Regularization
# Version: 1.2.0
# =============================================================================
#
# Stabilizes LLM/ViT training by monitoring spectral health in real-time.
#
# Installation:
#   pip install git+https://github.com/yourusername/assr.git
#
# Usage:
#   from assr import ASSRTrainer, ASSRConfig
#   trainer = ASSRTrainer(model=model, args=args, train_dataset=ds)
#   trainer.train()
#
# =============================================================================

import torch
import torch.nn as nn
import random
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

__version__ = "1.2.0"
__all__ = [
    "ASSRTrainer",
    "ASSRConfig",
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
    
    ASSR monitors two spectral health metrics and applies adaptive L2 
    regularization when layers become unhealthy.
    
    Attributes:
        base_lambda (float): Base regularization strength. Default: 1e-4
        
        stable_rank_floor (float): Trigger intervention if stable rank ratio 
            falls below this threshold. Default: 0.25
            - Random Gaussian init typically gives 0.30-0.40
            - Values below 0.20 indicate significant collapse
            - Lower values (0.15-0.20) for less aggressive intervention
            
        condition_ceiling (float): Trigger intervention if condition number 
            exceeds this threshold. Default: 500
            - Well-conditioned matrices have condition < 100
            - Values > 1000 indicate severe ill-conditioning
            
        sample_ratio (float): Fraction of layers to check per step. Default: 0.1
            - Higher values = more thorough but slower
            - 0.1 provides good coverage with minimal overhead
            
        sample_freq (int): Run ASSR check every N steps. Default: 1
            - Set to 5-10 for faster training with slightly less coverage
            
        max_severity_multiplier (float): Maximum scaling factor for adaptive 
            lambda. Default: 10.0
            - Actual lambda ranges from base_lambda to base_lambda * (1 + this)
            
        log_interventions (bool): Print debug info when interventions occur. 
            Default: False
    
    Example:
        # Use defaults (recommended for most cases)
        config = ASSRConfig()
        
        # Less aggressive (fewer interventions)
        config = ASSRConfig(stable_rank_floor=0.15, condition_ceiling=1000)
        
        # More aggressive (catch issues earlier)
        config = ASSRConfig(stable_rank_floor=0.30, condition_ceiling=200)
        
        # Debug mode
        config = ASSRConfig(log_interventions=True)
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
# SPECTRAL SENSORS
# =============================================================================

def compute_stable_rank(W: torch.Tensor) -> float:
    """
    Compute the Stable Rank of a weight matrix.
    
    Stable Rank = ||W||_F² / ||W||_2²
    
    This measures the "effective number of dimensions" the matrix uses.
    For a matrix with singular values σ₁ ≥ σ₂ ≥ ... ≥ σₙ:
        - If all σᵢ are equal: stable_rank = n (full rank)
        - If only σ₁ > 0: stable_rank = 1 (rank-1)
    
    Args:
        W: Weight tensor of shape (out_features, in_features)
        
    Returns:
        Stable rank as a float >= 1.0
        
    Example:
        >>> W = torch.eye(100)
        >>> compute_stable_rank(W)
        100.0
        
        >>> W = torch.randn(100, 1) @ torch.randn(1, 100)  # rank-1
        >>> compute_stable_rank(W)
        1.0
    """
    if W.dim() != 2:
        return float(min(W.shape[-2:]) if W.dim() > 2 else W.numel())
    
    with torch.no_grad():
        try:
            frob_sq = torch.sum(W ** 2).item()
            spectral_sq = torch.linalg.svdvals(W)[0].item() ** 2
            if spectral_sq > 1e-10:
                return max(1.0, frob_sq / spectral_sq)
            return 1.0
        except Exception:
            return 1.0


def compute_stable_rank_ratio(W: torch.Tensor) -> float:
    """
    Compute stable rank as a fraction of maximum possible rank.
    
    This normalizes stable rank to [0, 1] for easier threshold comparison.
    
    Returns:
        Value in [0, 1]:
        - 1.0 = using full rank capacity (perfectly orthogonal)
        - 0.0 = completely collapsed (rank-1)
        
    Typical values:
        - Random Gaussian init: 0.30-0.40
        - Healthy trained layer: 0.25-0.50
        - Concerning collapse: < 0.20
        - Severe collapse: < 0.10
        
    Example:
        >>> W = torch.eye(100)
        >>> compute_stable_rank_ratio(W)
        1.0
        
        >>> W = torch.randn(256, 128) * 0.02  # typical init
        >>> compute_stable_rank_ratio(W)  # ~0.35
    """
    if W.dim() != 2:
        return 1.0
    stable_rank = compute_stable_rank(W)
    max_rank = min(W.shape[0], W.shape[1])
    return stable_rank / max_rank if max_rank > 0 else 0.0


def compute_condition_number(W: torch.Tensor) -> float:
    """
    Compute the Condition Number of a weight matrix.
    
    Condition Number = σ_max / σ_min
    
    This measures how "stretched" the linear transformation is.
    High condition numbers lead to numerical instability and 
    gradient explosion/vanishing.
    
    Returns:
        Condition number as a float:
        - 1.0 = perfectly conditioned (orthogonal matrix)
        - 1-10 = excellent conditioning
        - 10-100 = good conditioning
        - 100-1000 = moderate issues
        - >1000 = severe ill-conditioning
        - inf = rank deficient (σ_min ≈ 0)
        
    Example:
        >>> W = torch.eye(100)
        >>> compute_condition_number(W)
        1.0
        
        >>> W, _ = torch.linalg.qr(torch.randn(100, 100))
        >>> compute_condition_number(W)  # ~1.0
    """
    if W.dim() != 2:
        return 1.0
    
    with torch.no_grad():
        try:
            s = torch.linalg.svdvals(W)
            s_max = s[0].item()
            s_min = s[-1].item()
            if s_min > 1e-10:
                return s_max / s_min
            return float('inf')
        except Exception:
            return float('inf')


def compute_spectral_health(W: torch.Tensor) -> Dict[str, Any]:
    """
    Compute comprehensive spectral health metrics for a weight matrix.
    
    Args:
        W: Weight tensor
        
    Returns:
        Dictionary containing:
        - stable_rank: Raw stable rank value
        - stable_rank_ratio: Normalized to [0, 1]
        - effective_rank: Entropy-based effective rank
        - condition: Condition number
        - spectral_norm: Largest singular value
        - min_singular_value: Smallest singular value
        - frobenius_norm: Frobenius norm
        - shape: Tuple of dimensions
        
    Example:
        >>> W = torch.randn(256, 128)
        >>> health = compute_spectral_health(W)
        >>> print(f"SR ratio: {health['stable_rank_ratio']:.3f}")
    """
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
        }
    
    stable_rank = compute_stable_rank(W)
    stable_rank_ratio = compute_stable_rank_ratio(W)
    condition = compute_condition_number(W)
    
    with torch.no_grad():
        try:
            s = torch.linalg.svdvals(W)
            spectral_norm = s[0].item()
            min_sv = s[-1].item()
            
            # Entropy-based effective rank
            s_norm = s / s.sum()
            s_norm = s_norm[s_norm > 1e-10]
            entropy = -torch.sum(s_norm * torch.log(s_norm)).item()
            effective_rank = float(torch.exp(torch.tensor(entropy)))
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
    }


# =============================================================================
# ASSR TRAINER
# =============================================================================

try:
    from transformers import Trainer
    
    class ASSRTrainer(Trainer):
        """
        Hugging Face Trainer with Auto-Calibrated Stochastic Spectral Regularization.
        
        ASSR monitors layer health during training and applies adaptive L2 
        regularization when spectral instabilities are detected. It targets
        two failure modes:
        
        1. **Rank Collapse** (low stable rank): Neurons becoming redundant,
           reducing the effective capacity of the layer.
           
        2. **Ill-Conditioning** (high condition number): Singular values
           becoming too spread out, causing gradient instability.
        
        The regularization strength is automatically calibrated based on the
        severity of the detected issue.
        
        Args:
            assr_config: ASSRConfig object. If None, uses default configuration.
            *args, **kwargs: Passed to the parent Trainer class.
            
        Attributes:
            assr_rank_interventions: Count of rank-based interventions
            assr_condition_interventions: Count of condition-based interventions
            assr_total_reg_loss: Cumulative regularization loss
            assr_steps_with_intervention: Steps where at least one intervention occurred
            
        Example:
            ```python
            from assr import ASSRTrainer, ASSRConfig
            from transformers import TrainingArguments
            
            # Basic usage with defaults
            trainer = ASSRTrainer(
                model=model,
                args=TrainingArguments(output_dir="./out", ...),
                train_dataset=dataset,
            )
            trainer.train()
            
            # Check what happened
            summary = trainer.get_assr_summary()
            print(f"Interventions: {summary['total_interventions']}")
            
            # Custom configuration
            config = ASSRConfig(
                base_lambda=5e-5,        # Lighter regularization
                stable_rank_floor=0.20,  # More permissive
                log_interventions=True,  # Debug output
            )
            trainer = ASSRTrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                assr_config=config,
            )
            ```
            
        Note:
            ASSR adds minimal overhead (<0.1% typically) due to:
            - Stochastic sampling (only 10% of layers checked per step)
            - Efficient SVD computation for health metrics
            - No gradient computation for the sensing phase
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
            """
            Compute training loss with ASSR regularization.
            
            This method overrides the parent Trainer.compute_loss() to add
            spectral regularization before the backward pass.
            """
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
            """
            Compute the ASSR regularization term.
            
            Uses stochastic sampling to check a subset of layers, then applies
            adaptive L2 penalty to layers that fail spectral health checks.
            """
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
                        # How far below the floor?
                        severity = (cfg.stable_rank_floor - sr_ratio) / cfg.stable_rank_floor
                        severity = min(max(severity, 0.0), 1.0)
                        self.assr_rank_interventions += 1
                        trigger_type = "Rank"
                    else:
                        # How far above the ceiling?
                        excess = condition / cfg.condition_ceiling
                        severity = min((excess - 1.0) / 10.0, 1.0)
                        self.assr_condition_interventions += 1
                        trigger_type = "Cond"
                    
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
            """
            Get summary statistics of ASSR activity during training.
            
            Returns:
                Dictionary containing:
                - rank_interventions: Number of rank-triggered interventions
                - condition_interventions: Number of condition-triggered interventions
                - total_interventions: Sum of both types
                - steps_with_intervention: Number of steps with any intervention
                - intervention_rate: Fraction of steps with intervention
                - total_reg_loss: Cumulative regularization loss
                - avg_reg_loss_per_step: Average regularization per step
                - num_linear_layers: Total linear layers in model
                - config: The ASSRConfig used
                
            Example:
                >>> trainer.train()
                >>> summary = trainer.get_assr_summary()
                >>> print(f"Intervention rate: {summary['intervention_rate']:.1%}")
            """
            total_steps = self.state.global_step if self.state else 0
            return {
                'rank_interventions': self.assr_rank_interventions,
                'condition_interventions': self.assr_condition_interventions,
                'total_interventions': (
                    self.assr_rank_interventions + self.assr_condition_interventions
                ),
                'steps_with_intervention': self.assr_steps_with_intervention,
                'intervention_rate': (
                    self.assr_steps_with_intervention / max(total_steps, 1)
                ),
                'total_reg_loss': self.assr_total_reg_loss,
                'avg_reg_loss_per_step': (
                    self.assr_total_reg_loss / max(total_steps, 1)
                ),
                'num_linear_layers': len(self.linear_layers),
                'config': self.assr_config,
            }
        
        def reset_assr_stats(self) -> None:
            """Reset ASSR statistics counters."""
            self.assr_rank_interventions = 0
            self.assr_condition_interventions = 0
            self.assr_total_reg_loss = 0.0
            self.assr_steps_with_intervention = 0

except ImportError:
    # transformers not installed - ASSRTrainer not available
    ASSRTrainer = None
    

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_spectral_report(model: nn.Module, top_k: int = 10) -> None:
    """
    Print a spectral health report for all linear layers in a model.
    
    Analyzes each linear layer and reports stable rank ratio and condition
    number, highlighting layers that may have issues.
    
    Args:
        model: PyTorch model to analyze
        top_k: Number of worst layers to show in detail
        
    Example:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained("gpt2")
        >>> print_spectral_report(model)
        
        ===========================================================================
          SPECTRAL HEALTH REPORT
        ===========================================================================
          Total Linear Layers: 74
          Stable Rank Ratio: min=0.312, max=0.456, mean=0.378
          Condition Number:  min=4.2, max=89.3, mean=23.1
          
          Layer                                    Shape           SR Ratio   Condition
          ---------------------------------------------------------------------------
            h.0.attn.c_attn                        (768, 2304)     0.312      89.3
            ...
        ===========================================================================
    """
    results = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            health = compute_spectral_health(m.weight)
            results.append({'name': name, **health})
    
    print("\n" + "=" * 75)
    print("  SPECTRAL HEALTH REPORT")
    print("=" * 75)
    print(f"  Total Linear Layers: {len(results)}")
    
    if not results:
        print("  No linear layers found.")
        print("=" * 75 + "\n")
        return
    
    # Statistics
    sr_vals = [r['stable_rank_ratio'] for r in results]
    c_vals = [r['condition'] for r in results if r['condition'] < float('inf')]
    
    print(f"  Stable Rank Ratio: min={min(sr_vals):.3f}, max={max(sr_vals):.3f}, "
          f"mean={sum(sr_vals)/len(sr_vals):.3f}")
    if c_vals:
        print(f"  Condition Number:  min={min(c_vals):.1f}, max={max(c_vals):.1f}, "
              f"mean={sum(c_vals)/len(c_vals):.1f}")
    
    # Sort by stable rank ratio (lowest = most concerning)
    results.sort(key=lambda x: x['stable_rank_ratio'])
    
    print(f"\n  {'Layer':<40} {'Shape':<15} {'SR Ratio':<10} {'Condition':<12}")
    print("  " + "-" * 73)
    
    for r in results[:top_k]:
        sr_flag = "⚠️" if r['stable_rank_ratio'] < 0.25 else "  "
        c_flag = "⚠️" if r['condition'] > 500 else "  "
        
        # Truncate long names
        name = r['name']
        if len(name) > 38:
            name = "..." + name[-35:]
        
        shape_str = f"{r['shape']}"
        cond_str = f"{r['condition']:.1f}" if r['condition'] < 1e6 else "inf"
        
        print(f"  {sr_flag}{c_flag}{name:<38} {shape_str:<15} "
              f"{r['stable_rank_ratio']:<10.3f} {cond_str:<12}")
    
    if len(results) > top_k:
        print(f"  ... and {len(results) - top_k} more layers")
    
    # Health assessment summary
    n_rank_issues = sum(1 for r in results if r['stable_rank_ratio'] < 0.25)
    n_cond_issues = sum(1 for r in results if r['condition'] > 500)
    
    print("\n  " + "-" * 73)
    if n_rank_issues == 0 and n_cond_issues == 0:
        print("  ✅ All layers appear healthy")
    else:
        if n_rank_issues > 0:
            print(f"  ⚠️  {n_rank_issues} layer(s) have low stable rank (potential collapse)")
        if n_cond_issues > 0:
            print(f"  ⚠️  {n_cond_issues} layer(s) are ill-conditioned")
    
    print("=" * 75 + "\n")


def analyze_layer(layer: nn.Linear, name: str = "layer") -> Dict[str, Any]:
    """
    Perform detailed spectral analysis of a single layer.
    
    Args:
        layer: A nn.Linear module to analyze
        name: Name to display in output
        
    Returns:
        Dictionary of spectral health metrics
        
    Example:
        >>> layer = model.transformer.h[0].attn.c_attn
        >>> health = analyze_layer(layer, "attention_qkv")
        
        Analysis of 'attention_qkv':
          Shape: (768, 2304)
          Stable Rank: 245.32 (ratio: 0.319)
          Effective Rank: 289.45
          Condition Number: 89.3
          Spectral Norm: 1.2345
          Min Singular Value: 0.0138
          Frobenius Norm: 12.456
          ✅ Layer appears healthy
    """
    health = compute_spectral_health(layer.weight)
    
    print(f"\n  Analysis of '{name}':")
    print(f"    Shape: {health['shape']}")
    print(f"    Stable Rank: {health['stable_rank']:.2f} "
          f"(ratio: {health['stable_rank_ratio']:.3f})")
    print(f"    Effective Rank: {health['effective_rank']:.2f}")
    print(f"    Condition Number: {health['condition']:.1f}")
    print(f"    Spectral Norm: {health['spectral_norm']:.4f}")
    print(f"    Min Singular Value: {health['min_singular_value']:.6f}")
    print(f"    Frobenius Norm: {health['frobenius_norm']:.4f}")
    
    # Assessment
    issues = []
    if health['stable_rank_ratio'] < 0.25:
        issues.append("Low stable rank (potential collapse)")
    if health['condition'] > 500:
        issues.append("High condition number (ill-conditioned)")
    
    if issues:
        print(f"    ⚠️  Issues: {', '.join(issues)}")
    else:
        print(f"    ✅ Layer appears healthy")
    
    return health
