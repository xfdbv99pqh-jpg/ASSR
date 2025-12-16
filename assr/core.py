import torch
import torch.nn as nn
import random
from transformers import TrainerCallback

def compute_fiedler_value(W):
    """
    Computes the Fiedler Value (Algebraic Connectivity) of a weight matrix.
    We treat the Gram matrix (W^T W) as the graph adjacency structure.
    
    The Fiedler value is the second smallest eigenvalue of the Laplacian.
    High Fiedler value = Strong neural connectivity (Healthy).
    Low Fiedler value = Spectral Collapse (Overfitting).
    """
    if W.dim() > 2: return 1.0 # Skip conv layers for now
    
    # 1. Construct the Gram Matrix (Correlation between neurons)
    # We detach to ensure this calculation doesn't explode the gradient graph
    with torch.no_grad():
        if W.shape[0] > W.shape[1]:
            G = torch.mm(W.t(), W)
        else:
            G = torch.mm(W, W.t())
            
        # 2. Construct the Combinatorial Laplacian (L = D - A)
        # Degree matrix D is diagonal of row sums
        D = torch.diag(torch.sum(torch.abs(G), dim=1))
        L = D - G
        
        # 3. Compute Eigenvalues (Symmetric eigendecomposition is stable)
        # We only need the smallest ones, but eigvalsh is fast on GPU
        try:
            eigvals = torch.linalg.eigvalsh(L)
            
            # The smallest eigenvalue is always 0 (for connected graphs)
            # The SECOND smallest is the Fiedler Value.
            if len(eigvals) > 1:
                fiedler = eigvals[1].item()
                return max(1e-6, fiedler) # Clamp to avoid division by zero
        except:
            return 1.0
            
    return 1.0

class ASSRCallback(TrainerCallback):
    """
    Auto-Calibrated Stochastic Spectral Regularization (ASSR)
    """
    def __init__(self, model, freq=10, ratio=0.1, base_lambda=0.001):
        self.freq = freq
        self.ratio = ratio
        self.base_lambda = base_lambda
        self.linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self.target_fiedler = 0.1 # We want to maintain at least this connectivity

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.freq != 0: return control
        
        # Stochastic Sampling: Efficiency Trick
        num_sample = max(1, int(len(self.linear_layers) * self.ratio))
        subset = random.sample(self.linear_layers, num_sample)
        
        for m in subset:
            if m.weight.grad is None: continue
            W = m.weight
            
            # --- AUTO-CALIBRATION LOGIC ---
            # 1. Measure the current health of this layer
            current_fiedler = compute_fiedler_value(W)
            
            # 2. Dynamic Lambda: If connectivity is low (collapse), INCREASE reg.
            # If connectivity is high (healthy), DECREASE reg.
            adaptive_lambda = self.base_lambda * (self.target_fiedler / current_fiedler)
            
            # --- SPECTRAL REGULARIZATION ---
            if W.shape[0] > W.shape[1]: 
                m_gram = torch.mm(W.t(), W); I = torch.eye(W.shape[1], device=W.device)
            else: 
                m_gram = torch.mm(W, W.t()); I = torch.eye(W.shape[0], device=W.device)
            
            # Apply the penalty
            loss = adaptive_lambda * (torch.norm(m_gram - I)**2 / m_gram.numel())
            loss.backward()
            
        return control
