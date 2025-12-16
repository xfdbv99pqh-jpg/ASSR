Here is the updated `README.md`. I have rewritten it to be transparent about the **Speed vs. Stability** trade-off we just discussed, while highlighting the massive **A100/1.1B Parameter** validation success.

---

#ASSR: Auto-Calibrated Stochastic Spectral Regularization**Robust Stability for Large Scale Model Training.**

ASSR is a PyTorch library that prevents training instability and rank collapse in Large Language Models (LLMs) and Vision Transformers. It replaces manual, static regularization with a **dynamic, closed-loop control system** that monitors the spectral health of your weight matrices in real-time.

**v1.3.0 Verified:** Successfully validated on **1.1B Parameter Llama** models on **NVIDIA A100 GPUs** using **BFloat16** precision.

##🚀 Why Use ASSR?Training massive models is expensive. A single instability spike on Day 3 can waste thousands of dollars in compute.

* **The Problem:** Standard regularization (Weight Decay) is "dumb"—it applies the same pressure to every layer, regardless of health.
* **The ASSR Solution:** ASSR acts like a thermostat. It scans layers for signs of collapse (Low Stable Rank) or explosion (High Condition Number) and applies surgical penalties only when needed.

##📊 Performance & Overhead**Does this slow down training?**
Yes, but it buys you reliability.

* **Throughput Cost:** Expect a **~15-25% reduction in steps-per-second** depending on your GPU and model size. Calculating SVDs and Eigenvalues for large matrices (e.g., 4096×4096) is mathematically heavy.
* **The Gain:**
* **Prevents Divergence:** Stops loss spikes before they kill a run.
* **No Manual Tuning:** The auto-calibrator finds the right thresholds for you.
* **Better Convergence:** Keeps weight matrices spectrally healthy, ensuring efficient gradient flow.



**Optimization:** We use **Stochastic Sampling** (checking only 10% of layers per step) to keep this overhead manageable.

##📦 InstallationInstall directly from the repository:

```bash
pip install git+https://github.com/xfdbv99pqh-jpg/ASSR.git

```

*Requires: `torch`, `transformers`, `numpy*`

##🛠️ Quick StartASSR drops into your existing Hugging Face training pipeline with zero friction.

###1. The "Set It and Forget It" Workflow (Recommended)For large models (Llama, Mistral, ViT-Large), use **Auto-Calibration**. This scans your specific model architecture to determine healthy spectral baselines.

```python
from transformers import AutoModelForCausalLM, TrainingArguments
from assr import ASSRTrainer, auto_calibrate

# 1. Load your model
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

# 2. Run Auto-Calibration
# This measures your model's initial health to set custom thresholds
config = auto_calibrate(model, verbose=True)

# 3. Use ASSRTrainer instead of the standard Trainer
trainer = ASSRTrainer(
    model=model,
    args=TrainingArguments(output_dir="checkpoints", learning_rate=2e-5),
    train_dataset=dataset,
    assr_config=config  # <--- Pass the calibrated config here
)

trainer.train()

```

###2. Manual ConfigurationFor research or smaller models, you can define the config manually:

```python
from assr import ASSRConfig, ASSRTrainer

config = ASSRConfig(
    base_lambda=1e-4,        # Strength of the L2 penalty
    stable_rank_floor=0.25,  # Trigger if rank drops below 25% of dimension
    condition_ceiling=1000,  # Trigger if condition number exceeds 1000
    sample_ratio=0.1,        # Check 10% of layers per step (Higher = Safer but Slower)
    sample_freq=1            # Check every N steps
)

trainer = ASSRTrainer(..., assr_config=config)

```

##🧠 How It Works: The Dual-Trigger SystemASSR uses a **Dual-Sensor Architecture** to detect two distinct types of failure:

1. **Sensor A: Stable Rank (The "Collapse" Detector)**
* *Math:* \frac{\|W\|_F^2}{\|W\|_2^2}
* *What it detects:* Neurons becoming redundant or "collapsing" into a lower dimension.
* *Action:* Triggers regularization to push neurons apart.


2. **Sensor B: Condition Number (The "Explosion" Detector)**
* *Math:* \frac{\sigma_{\max}}{\sigma_{\min}}
* *What it detects:* Exploding gradients or extreme sensitivity to noise.
* *Action:* Triggers regularization to constrain the largest singular values.



**The Actuator:**
When a layer triggers an alarm, ASSR applies a **Soft L2 Penalty** scaled by the *severity* of the violation. This gently nudges the layer back into the healthy zone without destroying previously learned features.

##🤝 Compatibility| Feature | Support | Note |
| --- | --- | --- |
| **Hugging Face** | ✅ Native | Drop-in replacement for `Trainer`. |
| **FP16 / BF16** | ✅ Native | **New in v1.3:** Sensors auto-cast to FP32 for stability on A100s. |
| **FSDP / DeepSpeed** | ⚠️ Partial | Sensors run on individual GPU shards (local rank). |

##LicenseMIT License. Free for research and commercial use.
