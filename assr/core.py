import torch
import torch.nn as nn
import random
from transformers import TrainerCallback

class ASSRCallback(TrainerCallback):
    def __init__(self, model, freq=10, ratio=0.1, lambda_mlp=0.001):
        self.freq = freq
        self.ratio = ratio
        self.lambda_mlp = lambda_mlp
        self.linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.freq != 0: return control

        # Stochastic Regularization
        num_sample = max(1, int(len(self.linear_layers) * self.ratio))
        subset = random.sample(self.linear_layers, num_sample)

        for m in subset:
            if m.weight.grad is None: continue
            W = m.weight
            # Efficient Gram Matrix
            if W.shape[0] > W.shape[1]: 
                m_gram = torch.mm(W.t(), W); I = torch.eye(W.shape[1], device=W.device)
            else: 
                m_gram = torch.mm(W, W.t()); I = torch.eye(W.shape[0], device=W.device)

            loss = self.lambda_mlp * (torch.norm(m_gram - I)**2 / m_gram.numel())
            loss.backward()
        return control
