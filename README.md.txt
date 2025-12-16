# ASSR: Auto-Calibrated Stochastic Spectral Regularization

**A self-tuning, compute-efficient spectral regularization library for Deep Learning.**

ASSR stabilizes neural network training by maintaining the **spectral health** of weight matrices. Unlike standard methods, ASSR is:

1.  **Auto-Calibrating:** Automatically adjusts regularization strength.
2.  **Stochastic:** Runs on <10% of layers to save compute.
3.  **Transformer-Ready:** Prevents "Head Collapse" in LLMs.

## Installation
Download the repository and import `ASSRCallback`.

## Quick Start
```python
from assr import ASSRCallback
# Add to your Hugging Face Trainer callbacks