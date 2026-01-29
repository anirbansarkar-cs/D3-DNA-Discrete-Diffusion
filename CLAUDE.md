# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

D3-DNA-Discrete-Diffusion implements discrete diffusion models for DNA sequence generation with tunable regulatory activity. The codebase uses **Score Entropy Discrete Diffusion (SEDD)** to generate biologically relevant DNA sequences conditioned on target regulatory activities.

## Essential Commands

### Installation
```bash
# Install in development mode (recommended)
pip install -e .

# Install with dev dependencies (pytest, black, flake8, mypy)
pip install -e ".[dev]"
```

### Linting and Formatting
```bash
# Format code
black .

# Check style
flake8

# Type checking
mypy
```

### Training
```bash
# Train with specific architecture (transformer or convolutional)
python model_zoo/deepstarr/train.py --architecture transformer
python model_zoo/mpra/train.py --architecture convolutional
python model_zoo/promoter/train.py --architecture transformer

# Resume from checkpoint
python model_zoo/deepstarr/train.py --architecture transformer --resume_from /path/to/checkpoint.ckpt

# With WandB logging
python model_zoo/deepstarr/train.py --architecture transformer --wandb_project d3-deepstarr --wandb_name exp1
```

### Sampling/Generation
```bash
# Basic sampling (unconditional)
python model_zoo/deepstarr/sample.py --architecture transformer --checkpoint model.ckpt --num_samples 1000

# Conditional generation with target activities
python model_zoo/deepstarr/sample.py --architecture transformer --checkpoint model.ckpt \
  --dev_activity 2.0 --hk_activity 1.5 --num_samples 1000

# Promoter generation with expression target
python model_zoo/promoter/sample.py --architecture transformer --checkpoint model.ckpt \
  --expression_target 3.0
```

### Evaluation
```bash
# Evaluate with oracle model for SP-MSE
python model_zoo/deepstarr/evaluate.py --architecture transformer \
  --checkpoint model.ckpt --use_oracle --oracle_checkpoint oracle.ckpt

# Basic evaluation (diffusion metrics only)
python model_zoo/deepstarr/evaluate.py --architecture transformer --checkpoint model.ckpt
```

## Code Architecture

### Three-Tier Inheritance Pattern

**Tier 1: Base Classes (scripts/)**
- `scripts/train.py`: `BaseD3LightningModule`, `BaseD3DataModule`, `BaseTrainer` - PyTorch Lightning training infrastructure
- `scripts/sample.py`: `BaseSampler` - Predictor-Corrector (PC) sampling framework with batching and memory management
- `scripts/evaluate.py`: `BaseEvaluator` - Oracle-based evaluation framework
- `scripts/sampling.py`: `Predictor` classes (`EulerPredictor`, `AnalyticPredictor`, `Denoiser`)

**Tier 2: Dataset-Specific (model_zoo/[dataset]/)**
Each dataset folder contains:
- `train.py`: Inherits from `BaseTrainer`, implements `create_lightning_module()` and `create_data_module()`
- `sample.py`: Inherits from `BaseSampler`, implements `load_model()`, `create_dataloader()`, `generate_conditioning_labels()`
- `evaluate.py`: Inherits from `BaseEvaluator`, implements oracle model loading
- `models.py`: Dataset-specific wrappers around core architectures (minimal dataset-specific preprocessing)
- `data.py`: Dataset loading logic
- `configs/`: Architecture-specific YAML configs (`transformer.yaml`, `convolutional.yaml`)

**Tier 3: Pure Architecture (model/)**
- `transformer.py`: `TransformerModel` - Pure transformer with rotary embeddings, flash attention
- `cnn.py`: `ConvolutionalModel` - Pure convolutional with dilated convolutions
- `layers.py`: Shared components (`TimestepEmbedder`, `LabelEmbedder`, etc.)
- **Key**: Models are completely dataset-agnostic; all dataset info passed via config

### Diffusion Model Implementation

**Forward Process (Noising)**:
- **Graph** (`utils/graph_lib.py`): Defines discrete transition dynamics
  - `Uniform`: Symmetric transitions between all tokens (A↔C↔G↔T)
  - `Absorbing`: Asymmetric transitions to special mask token (A→M, C→M, etc.)
  - Methods: `rate()`, `transition()`, `sample_transition()`, `reverse_rate()`, `staggered_score()`
- **Noise Schedule** (`utils/noise_lib.py`): `GeometricNoise` implements σ(t) = σ_min^(1-t) * σ_max^t

**Reverse Process (Denoising)**:
- **Predictor-Corrector Sampling** (`scripts/sampling.py`):
  - `EulerPredictor`: Euler method with reverse rate
  - `AnalyticPredictor`: Uses staggered score and Tweedie's formula
  - `Denoiser`: Final denoising at t=0
- **Score Function**: Model outputs log-scores (training) → exponentiated to scores (sampling)

**Loss Function** (`utils/losses.py`):
- Score entropy: `graph.score_entropy(log_score, sigma, perturbed_batch, batch)`
- Weighted by noise rate: `(dsigma * loss).sum()`
- Optional sampling consistency (SC) loss

### Conditional Generation

**Three Conditioning Mechanisms**:

1. **Global Label Conditioning**: Labels broadcast to all positions
   ```python
   signal_embed = self.signal_embedding(labels)  # (batch, signal_dim) → (batch, embed_dim)
   vocab_embed = self.embedding[x]               # (batch, seq_len, embed_dim)
   result = vocab_embed + signal_embed[:, None, :]  # Broadcast
   ```

2. **Time/Noise Conditioning**: Always present, modulates all layers
   ```python
   sigma_embed = self.sigma_map(sigma)  # Sinusoidal time embedding
   c = F.silu(sigma_embed)              # Modulation signal for layers
   ```

3. **Classifier-Free Guidance**: `LabelEmbedder` supports label dropout for unconditional generation

**Transformer**: Labels → Linear embedding → Added to sequence embeddings, Time → AdaLN modulation
**Convolutional**: Labels → Preprocessed features → Concatenated with one-hot, Time → Channel addition

### SP-MSE Validation System

**Purpose**: Evaluate biological relevance of generated sequences during training

**Architecture** (`utils/sp_mse_callback.py`):
- `BaseSPMSEValidationCallback`: PyTorch Lightning callback
- Loads pre-trained oracle model (e.g., DeepSTARR predictor)
- Samples sequences at validation epochs using PC sampler with EMA weights
- Computes SP-MSE: `mean((oracle(real_data) - oracle(generated))²)`
- Implements early stopping based on lowest SP-MSE

**Oracle Models**:
- DeepSTARR: `PL_DeepSTARR` convolutional predictor (`model_zoo/deepstarr/deepstarr.py`)
- MPRA: `PL_mpra` predictor
- Promoter: SEI (Sequence-to-Expression and Interaction) model

### Configuration System

Each dataset has architecture-specific configs in `model_zoo/[dataset]/configs/`:

```yaml
dataset:
  name: deepstarr
  sequence_length: 249
  num_classes: 4      # DNA vocabulary size (A, C, G, T)
  signal_dim: 2       # Conditioning dimension (dev + hk activities)

model:
  architecture: transformer  # or convolutional
  hidden_size: 768
  n_blocks: 12
  n_heads: 12
  cond_dim: 128      # Time embedding dimension

graph:
  type: uniform      # or absorb

noise:
  type: geometric
  sigma_min: 0.0001
  sigma_max: 20

training:
  batch_size: 256
  n_iters: 1000000
  lr: 0.0003
```

**Key Differences Between Architectures**:
- Transformer: `cond_dim: 128`, rotary embeddings, flash attention
- Convolutional: `cond_dim: 256`, dilated convolutions [1,1,4,16,64]×4, group normalization

## Important Implementation Details

### Score vs Log-Score Convention
- **Training**: Model outputs log-scores for numerical stability
- **Sampling**: Scores exponentiated: `score_fn(...).exp()`
- **Loss**: Expects log-scores: `graph.score_entropy(log_score, ...)`

### Two-Stage Model Loading
Lightning checkpoints use prefixed keys (`score_model.*`, `ema.*`), while original checkpoints use dict format (`{'model': ..., 'ema': ..., 'step': ...}`). The `load_state_dict()` method automatically detects the format.

### Batched Sampling with Memory Management
`BaseSampler` automatically batches large jobs (>512 samples), moves tensors to CPU after each batch to free GPU memory, and consolidates saved elements efficiently.

### Exponential Moving Average (EMA)
- Maintained during training: `shadow = decay * shadow + (1-decay) * param`
- Dynamic decay: `min(decay, (1+num_updates)/(10+num_updates))`
- **Always use EMA weights for validation and sampling** for better stability

### Flash Attention Integration
Uses `flash_attn_qkvpacked_func` for fixed-length sequences. **Critical**: QKV must be contiguous for H100 GPUs. Fallback to `flash_attn_varlen_qkvpacked_func` for variable-length.

### Gradient Accumulation
- Effective batch size = `batch_size / (ngpus * accum)`
- Loss scaled by `1/accum` before backward
- EMA updates only after full accumulation cycle

### Staggered Score Computation
Implements Tweedie's formula via matrix exponential approximation. Different formulas for Uniform vs Absorbing graphs. Computes p_{σ-Δσ}(z) / p_σ(x).

## Adding New Datasets

The modular architecture requires minimal changes:

1. **Create dataset directory**: `mkdir model_zoo/my_dataset`

2. **Implement required files**:
   - `data.py`: Dataset loading (implement `get_my_dataset_datasets()`)
   - `models.py`: Model factory (implement `create_model(config, architecture)`)
   - `train.py`: Inherit from `BaseTrainer`
   - `sample.py`: Inherit from `BaseSampler`
   - `evaluate.py`: Inherit from `BaseEvaluator`
   - `configs/transformer.yaml` and `configs/convolutional.yaml`

3. **No changes to core codebase needed** - all dataset-specific logic is isolated

## Key Utilities (utils/)

- **graph_lib.py**: Discrete diffusion graphs (`Uniform`, `Absorbing`)
- **noise_lib.py**: Noise schedules (`GeometricNoise`)
- **losses.py**: Score entropy loss, optimizer setup
- **catsample.py**: Categorical sampling with Gumbel trick
- **data_utils.py**: `cycle_loader()`, `one_hot_encode_sequences()`, `reverse_complement()`, `calculate_gc_content()`
- **sp_mse_callback.py**: Oracle-based validation callbacks
- **inpainting.py**: `InpaintingManager` for constrained generation

## WandB Integration

Training and sampling scripts support Weights & Biases for experiment tracking:

```bash
# Training with WandB
python model_zoo/deepstarr/train.py --architecture transformer \
  --wandb_project d3-deepstarr --wandb_name experiment-1

# Sampling with WandB
python model_zoo/deepstarr/sample.py --architecture transformer --checkpoint model.ckpt \
  --use_wandb --wandb_project d3-deepstarr-sampling --wandb_name baseline-run
```

Project naming convention: `d3-{dataset}-{experiment_type}` (e.g., `d3-deepstarr-sampling`, `d3-lentimpra-conditional`)

## Datasets

### DeepSTARR
- **Sequence Length**: 249 bp
- **Labels**: 2 (developmental + housekeeping enhancer activities)
- **Data**: `model_zoo/deepstarr/DeepSTARR_data.h5`
- **Oracle**: PL_DeepSTARR convolutional predictor

### MPRA
- **Sequence Length**: 200 bp
- **Labels**: 3 (regulatory activity measurements)
- **Data**: `model_zoo/mpra/MPRA_data.h5`
- **Oracle**: PL_mpra predictor

### Promoter
- **Sequence Length**: 1024 bp
- **Labels**: Expression values (concatenated with sequences)
- **Oracle**: SEI (Sequence-to-Expression and Interaction) model

### LentiMPRA
- **Sequence Length**: 230 bp
- **Labels**: Cell line-specific regulatory activities (K562, HepG2, WTC11)
- **Variants**: Single-class and multi-class (3 cell lines combined)
- **Oracle**: Cell line-specific predictors

## Development Principles

1. **Base classes provide shared functionality** - Avoid duplicating training/sampling logic
2. **Dataset-specific classes inherit and customize** - Only override what's different
3. **No hardcoded dataset logic in shared components** - Use configs
4. **Pure architectures in model/** - Completely dataset-agnostic
5. **Configuration-driven behavior** - Change model behavior via YAML, not code
