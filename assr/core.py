import torch
import torch.nn as nn
from transformers import TrainerCallback
import random
import math
from typing import List, Union

# =============================================================================
# CORE SPECTRAL SENSORS (NO GRADIENT)
# =============================================================================

def compute_fiedler_value(W: torch.Tensor) -> float:
    """
    Sensor 1: Computes the Fiedler Value (Algebraic Connectivity).
    Measures structural integrity; low value indicates clustered/redundant neurons.
    """
    if W.dim() > 2: return 1.0
    
    with torch.no_grad():
        if W.shape[0] > W.shape[1]: G = torch.mm(W.t(), W)
        else: G = torch.mm(W, W.t())
        
        # Laplacian L = D - G
        D = torch.diag(torch.sum(torch.abs(G), dim=1))
        L = D - G
        
        try:
            # Use eigvalsh for fast, stable symmetric decomposition
            eigvals = torch.linalg.eigvalsh(L)
            if len(eigvals) > 1:
                # The Fiedler value is the second smallest eigenvalue
                return max(1e-9, eigvals[1].item())
        except:
            return 1.0
            
    return 1.0

def compute_condition_number(W: torch.Tensor) -> float:
    """
    Sensor 2 (Fail-Safe): Computes the Condition Number (sigma_max / sigma_min).
    Measures ill-conditioning; high value indicates gradient explosion risk.
    """
    with torch.no_grad():
        try:
            s = torch.linalg.svdvals(W)
            s_min = s[-1].item()
            s_max = s[0].item()
            # Return high value if rank is effectively zero
            return s_max / s_min if s_min > 1e-10 else float('inf')
        except:
            return float('inf')

# =============================================================================
# AUTO-CALIBRATED SPECTRAL REGULARIZATION CALLBACK
# =============================================================================

class ASSRCallback(TrainerCallback):
    """
    Auto-Calibrated Stochastic Spectral Regularization (ASSR).
    Uses a Dual-Trigger sensor to activate a stable L2 penalty.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 freq: int = 10, 
                 ratio: float = 0.1, 
                 base_lambda: float = 2e-5,
                 fiedler_floor: float = 0.05,
                 condition_ceiling: float = 1000.0,
                 ):
        
        self.freq = freq
        self.ratio = ratio
        self.base_lambda = base_lambda
        self.fiedler_floor = fiedler_floor
        self.condition_ceiling = condition_ceiling
        
        # Only target nn.Linear layers for spectral analysis
        self.linear_layers: List[nn.Module] = [
            m for m in model.modules() if isinstance(m, nn.Linear)
        ]
        
    def on_step_end(self, args, state, control, **kwargs):
        """
        Main hook: Runs after gradients are computed but before optimizer step.
        """
        if state.global_step % self.freq != 0: return control
        
        # Stochastic Sampling for low overhead
        num_sample = max(1, int(len(self.linear_layers) * self.ratio))
        subset = random.sample(self.linear_layers, num_sample)
        
        reg_loss = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        for m in subset:
            if m.weight.grad is None: continue
            W = m.weight
            
            # --- DUAL TRIGGER SENSOR ---
            fiedler = compute_fiedler_value(W)
            condition = compute_condition_number(W)
            
            is_fiedler_low = fiedler < self.fiedler_floor
            is_condition_high = condition > self.condition_ceiling
            
            if is_fiedler_low or is_condition_high:
                
                # 1. Determine severity for adaptive lambda
                if is_fiedler_low:
                    # Severity based on Fiedler metric
                    severity = (self.fiedler_floor - fiedler) / self.fiedler_floor
                else: 
                    # If Condition triggers, treat it as max initial severity
                    severity = 1.0 
                
                # 2. Adaptive Lambda: Scale the L2 weight decay strength
                # Multiplier (1 + 10*severity) means lambda is between base_lambda and 11*base_lambda
                adaptive_lambda = self.base_lambda * (1 + 10 * severity)
                
                # 3. STABLE L2 ACTUATOR
                # The stable penalty: L2 Norm (Frobenius Norm)
                penalty = torch.norm(W)**2 
                reg_loss = reg_loss + adaptive_lambda * penalty
        
        # Perform the backward pass for the regularization term
        if reg_loss.item() > 1e-8:
             reg_loss.backward()
             
        return control

# =============================================================================
# END OF LIBRARY CODE
# =============================================================================
