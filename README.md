# D3-DNA-Discrete-Diffusion

This repo contains a PyTorch implementation for the paper "Designing DNA With Tunable Regulatory Activity Using Discrete Diffusion". The training and sampling part of the code is inspired by [Score entropy discrete diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).


## 📁 Project Structure

```
D3-DNA-Discrete-Diffusion/
├── scripts/                                # Base classes + entry-point scripts
│   ├── train.py                            # Base training (PyTorch Lightning)
│   ├── evaluate.py                         # Base evaluation with common metrics
│   ├── sample.py                           # Base sampling (Predictor-Corrector)
│   ├── sampling.py                         # Predictor / Denoiser implementations
│   ├── deepstarr_attributions_template.py  # Oracle-check + DeepLIFT-SHAP template
│   └── shuffled_evaluations.py             # Dinuc-shuffle baseline evaluations
├── model_zoo/                              # Dataset-specific implementations
│   ├── deepstarr/                          # DeepSTARR enhancers (249 bp)
│   ├── lentimpra/                          # LentiMPRA K562/HepG2/WTC11 (230 bp)
│   ├── mpra/                               # MPRA (200 bp)
│   ├── promoter/                           # Promoter w/ expression (1024 bp)
│   ├── atacseq/                            # ATAC-seq
│   └── ccre/                               # cCRE
├── utils/                                  # Shared utilities
│   ├── data_utils.py                       # cycle_loader, one-hot, reverse-complement, ...
│   ├── graph_lib.py                        # Discrete diffusion graphs (Uniform / Absorbing)
│   ├── noise_lib.py                        # GeometricNoise schedule
│   ├── losses.py                           # Score-entropy loss, optimizer setup
│   ├── inpainting.py                       # InpaintingManager for constrained generation
│   ├── dinuc_shuffle.py                    # Dinucleotide-preserving shuffle
│   ├── sp_mse_callback.py                  # Base oracle-based validation callback
│   └── catsample.py                        # Categorical Gumbel sampling
├── model/                                  # Pure architectures (dataset-agnostic)
│   ├── transformer.py                      # Transformer w/ rotary + flash-attn
│   ├── cnn.py                              # Convolutional w/ dilated convs
│   ├── rotary.py                           # Rotary embeddings
│   └── layers.py                           # TimestepEmbedder, LabelEmbedder, ...
├── environment_new.yml                     # Conda environment (`d3-new`)
├── post_install.sh                         # Pip steps yml can't express (torch cu121, flash-attn, -e install)
├── setup_new.sh                            # Wrapper that runs the two steps above
└── pyproject.toml                          # Package configuration
```

## ⚡ Quick Start

### Installation

The environment is split between a conda yml (system + pure-pip packages) and a tiny post-install script (the things yml can't express: torch from the cu121 index URL, flash-attn with `--no-build-isolation`, and the editable install of this repo):

```bash
git clone https://github.com/anirbansarkar-cs/D3-DNA-Discrete-Diffusion.git
cd D3-DNA-Discrete-Diffusion

conda env create -f environment_new.yml -n d3-new
conda activate d3-new
bash post_install.sh
```

`setup_new.sh` wraps all three steps if you'd rather run a single command.

After install, `d3-new` includes:
- PyTorch 2.5.1 (cu121 wheels, ships its own CUDA 12 runtime), flash-attn 2.7.4.post1 (prebuilt wheel — no source build)
- PyTorch Lightning, einops, h5py, omegaconf, wandb, matplotlib, seaborn, scikit-learn
- evoaug2 (data augmentation: `RobustLoader`, `RandomRC`, `RandomMutation`, ...)
- fsspec / pyfaidx / s3fs (for S3 FASTA access in the cCRE benchmark)

#### Optional: TangerMeme (for `scripts/deepstarr_attributions_template.py`)

The attribution template depends on a TangerMeme fix that's currently an open PR upstream. Install from the PR branch until it lands:

```bash
pip install -e ~/tangermeme-worktrees/add-hook-stacks
```

### Training Models
```bash
# Train DeepSTARR with transformer
python model_zoo/deepstarr/train.py --architecture transformer

# Train LentiMPRA with convolutional architecture
python model_zoo/lentimpra/train.py --architecture convolutional

# Train Promoter with custom config
python model_zoo/promoter/train.py --architecture transformer --config custom.yaml

# Multi-GPU (4 GPUs)
python model_zoo/deepstarr/train.py --architecture transformer --ngpus 4

# Resume from checkpoint
python model_zoo/deepstarr/train.py --architecture transformer --resume /path/to/checkpoint.ckpt
```

### Evaluation
```bash
# Diffusion metrics only
python model_zoo/deepstarr/evaluate.py --architecture transformer --checkpoint model.ckpt

# With oracle (SP-MSE)
python model_zoo/deepstarr/evaluate.py --architecture transformer \
    --checkpoint model.ckpt --use_oracle --oracle_checkpoint oracle.ckpt
```

### Sampling

The base `scripts/sample.py` defines all shared CLI args (`--checkpoint`, `--architecture`, `--num_samples`, `--steps`, `--batch_size`, `--output`, `--format`, `--initial_condition`, `--start_at_timestep`, `--inpainting_*`, `--use_wandb`, ...). Each dataset's `model_zoo/<dataset>/sample.py` adds its own conditioning args.

```bash
# Unconditional DeepSTARR
python model_zoo/deepstarr/sample.py \
    --architecture transformer --checkpoint model.ckpt \
    --num_samples 1000 --unconditional

# DeepSTARR conditioned on dev + hk activities
python model_zoo/deepstarr/sample.py \
    --architecture transformer --checkpoint model.ckpt \
    --num_samples 1000 --dev_activity 2.0 --hk_activity 1.5

# LentiMPRA single-cell-line (K562)
python model_zoo/lentimpra/sample.py \
    --architecture transformer --checkpoint model.ckpt \
    --num_samples 1000 --activity 1.5

# LentiMPRA multi-class (K562 / HepG2 / WTC11)
python model_zoo/lentimpra/sample.py \
    --architecture transformer --checkpoint model.ckpt \
    --num_samples 1000 --k562_activity 2.0 --hepg2_activity 1.0 --wtc11_activity 0.5

# Promoter conditioned on expression
python model_zoo/promoter/sample.py \
    --architecture transformer --checkpoint model.ckpt \
    --num_samples 1000 --expression_target 3.0
```

#### Log a sampling run to WandB

```bash
python model_zoo/deepstarr/sample.py \
    --architecture transformer --checkpoint model.ckpt \
    --num_samples 1000 --dev_activity 2.0 --hk_activity 1.5 \
    --use_wandb --wandb_project d3-deepstarr-sampling --wandb_name dev2.0-hk1.5
```

#### Custom initial conditions / delayed sampling

```bash
# Start from a custom set of sequences in an H5 file, beginning at timestep 25
python model_zoo/lentimpra/sample.py \
    --architecture transformer --checkpoint model.ckpt \
    --num_samples 1000 --activity 1.5 \
    --custom_inits_path /path/to/seed_seqs.h5 --custom_inits_step 25

# Initialize from nucleotide frequencies and start the reverse process at timestep 10
python model_zoo/deepstarr/sample.py \
    --architecture transformer --checkpoint model.ckpt \
    --num_samples 1000 --unconditional \
    --initial_condition nuc_freq --start_at_timestep 10
```

### Motif Inpainting

`utils/inpainting.py` implements an `InpaintingManager` that constrains the reverse-diffusion process. Run sampling with `--inpainting_mode {inpaint_motifs, inpaint_not_motifs}` and an H5 of pre-extracted motif spans:

```bash
# Keep motif spans fixed, regenerate flanks
python model_zoo/deepstarr/sample.py \
    --architecture transformer --checkpoint model.ckpt \
    --num_samples 100 --dev_activity 2.0 --hk_activity 1.5 \
    --inpainting_mode inpaint_motifs \
    --inpainting_data inpainting_data/all_hits_combined.h5 \
    --inpainting_iterations 5

# Inverse: keep flanks, regenerate motif spans
python model_zoo/deepstarr/sample.py \
    --architecture transformer --checkpoint model.ckpt \
    --num_samples 100 --dev_activity 2.0 --hk_activity 1.5 \
    --inpainting_mode inpaint_not_motifs \
    --inpainting_data inpainting_data/all_hits_combined.h5
```

### Attribution maps (DeepSTARR)

`scripts/deepstarr_attributions_template.py` is a template script with two sub-commands. The `attributions` subcommand requires the TangerMeme PR fix (see Installation, above).

```bash
# Sanity-check the oracle on originals vs dinuc-shuffled motif/flank segments
python scripts/deepstarr_attributions_template.py oracle-check \
    --data_path samples.h5 --ckpt_path oracle.ckpt --mode motif

# DeepLIFT-SHAP attributions for dev + hk heads, per-timestep
python scripts/deepstarr_attributions_template.py attributions \
    --h5-samples samples.h5 --h5-config config.h5 \
    --deepstarr-ckpt oracle.ckpt --output-file attributions.h5
```

## 🧬 Supported Datasets

### DeepSTARR
- **Purpose**: Enhancer activity prediction
- **Sequence Length**: 249 bp
- **Labels**: 2 (developmental + housekeeping enhancer activities)
- **Oracle**: `PL_DeepSTARR` convolutional predictor for SP-MSE

### LentiMPRA
- **Purpose**: Regulatory activity in lentiviral MPRA across cell lines
- **Sequence Length**: 230 bp
- **Labels**: 1 (K562) or 3 (K562 + HepG2 + WTC11)
- **Oracle**: LegNet (`model_zoo/lentimpra/mpralegnet.py`)

### MPRA
- **Purpose**: Regulatory sequence analysis
- **Sequence Length**: 200 bp
- **Labels**: 3 (regulatory activity measurements)
- **Oracle**: `PL_mpra` predictor for SP-MSE

### Promoter
- **Purpose**: Gene expression prediction
- **Sequence Length**: 1024 bp
- **Labels**: Expression values (concatenated with sequences)
- **Oracle**: SEI (Sequence-to-Expression and Interaction) model

### ATAC-seq
- **Purpose**: Chromatin-accessibility-derived regulatory sequence modeling
- **Available** in `model_zoo/atacseq/` (train/evaluate/sample)

### cCRE
- **Purpose**: Candidate cis-Regulatory Element modeling
- **Available** in `model_zoo/ccre/` (train/evaluate/sample); `traitgym_benchmark.py` for downstream TraitGym evaluation

## 🔧 Model Architectures

### Transformer
- Multi-head attention with rotary positional embeddings
- Flash-attention (flash-attn 2.7.x) for fixed-length sequences
- Layer normalization, AdaLN time-modulation, residual connections
- Conditional generation via additive label embeddings
- Configurable depth (`n_blocks`) and width (`hidden_size`)

### Convolutional
- Multi-scale convolutional layers with dilated convolutions (`[1,1,4,16,64]×4`)
- Group normalization, residual connections
- Adaptive pooling for variable-length inputs
- Conditional generation via concatenated label features

## 📊 Advanced Features

### Sampling
- **Predictor-Corrector** with `EulerPredictor`, `AnalyticPredictor` (Tweedie's formula), and a final `Denoiser` at t=0 — see `scripts/sampling.py`.
- **Conditional generation** via global label conditioning (additive on transformer, concatenative on CNN).
- **Custom initial conditions**: load arbitrary starting sequences (`--custom_inits_path`), or initialize from nucleotide frequencies (`--initial_condition nuc_freq`).
- **Delayed sampling**: start the reverse process partway through (`--start_at_timestep N`).
- **Motif inpainting**: `--inpainting_mode inpaint_motifs` / `inpaint_not_motifs` with an H5 of motif spans.
- **WandB logging** at sampling time (sequences, oracle activities, artifacts) via `--use_wandb`.

### SP-MSE Validation
Evaluate biological relevance during training using oracle models:
```yaml
# In dataset config
sp_mse_validation:
  enabled: true
  validation_freq: 5000
  validation_samples: 1000
  early_stopping_patience: 3
```

### Multi-GPU Training
```yaml
ngpus: 4
nnodes: 1
training:
  batch_size: 1024
  accum: 1
```

## 🆕 Adding New Datasets

The modular architecture makes adding datasets simple:

1. **Create dataset directory**:
   ```bash
   mkdir model_zoo/my_dataset
   ```

2. **Implement required files**:
   ```python
   # model_zoo/my_dataset/data.py
   def get_my_dataset_datasets():
       # Dataset loading logic
       pass

   # model_zoo/my_dataset/models.py
   def create_model(config, architecture):
       # Model creation logic
       pass
   ```

3. **Create training script**:
   ```python
   # model_zoo/my_dataset/train.py
   from scripts.train import BaseTrainer

   class MyDatasetTrainer(BaseTrainer):
       # Inherit shared functionality
       pass
   ```

4. **Add configs**: Place YAML files in `model_zoo/my_dataset/configs/`

**That's it!** No changes to core codebase needed.

## 📋 Configuration

Each dataset has architecture-specific configs:
```yaml
# model_zoo/deepstarr/configs/transformer.yaml
dataset:
  name: deepstarr
  data_file: model_zoo/deepstarr/DeepSTARR_data.h5
  sequence_length: 249

model:
  architecture: transformer
  hidden_size: 768
  n_blocks: 12
  n_heads: 12

training:
  batch_size: 256
  n_iters: 1000000
  lr: 0.0003
```

## 📈 Results & Evaluation

Our models achieve state-of-the-art performance:
- **DeepSTARR**: High correlation with enhancer activities
- **MPRA / LentiMPRA**: Accurate regulatory predictions
- **Promoter**: Precise expression control

Evaluation metrics include:
- SP-MSE (oracle-based biological relevance)
- Standard diffusion metrics (loss, perplexity)
- Dataset-specific biological metrics

## 🛠️ Development

### Architecture Principles
1. **Base classes** provide shared functionality
2. **Dataset-specific classes** inherit and customize
3. **No hardcoded dataset logic** in shared components
4. **Configuration-driven** behavior
5. **Clean separation** of concerns

### Code Quality
- Fully type-hinted Python
- Comprehensive docstrings
- Modular, testable design
- Professional packaging with `pyproject.toml`

## 📚 Resources

### Datasets
- [DeepSTARR](https://huggingface.co/datasets/anonymous-3E42/DeepSTARR_preprocessed)
- [MPRA](https://huggingface.co/datasets/anonymous-3E42/MPRA_preprocessed)

### Oracle Models
- [DeepSTARR Oracle](https://huggingface.co/anonymous-3E42/DeepSTARR_oracle)
- [MPRA Oracle](https://huggingface.co/anonymous-3E42/MPRA_oracle)

### Pre-trained Models
- [DeepSTARR Transformer](https://huggingface.co/anonymous-3E42/DeepSTARR_D3_Tran_model)
- [DeepSTARR Convolutional](https://huggingface.co/anonymous-3E42/DeepSTARR_D3_Conv_model)
- [MPRA Transformer](https://huggingface.co/anonymous-3E42/MPRA_D3_Tran_model)
- [MPRA Convolutional](https://huggingface.co/anonymous-3E42/MPRA_D3_Conv_model)
- [Promoter Transformer](https://huggingface.co/anonymous-3E42/Promoter_D3_Tran_model)
- [Promoter Convolutional](https://huggingface.co/anonymous-3E42/Promoter_D3_Conv_model)

## 📜 Citation

```bibtex
@article{sarkar2024d3dna,
    title = {Designing {DNA} With Tunable Regulatory Activity Using Discrete Diffusion},
    author = {Sarkar, Anirban and Duran, Alejandra and Yu, Yiyang and Lin, Da-Wei and Kang, Yijie and Somia, Nirali and Mantilla, Pablo and Zhou, Jessica and Nagai, Masayuki and Tang, Ziqi and Hanington, Kaarina and Chang, Kenneth and Koo, Peter K.},
    journal = {bioRxiv},
    year = {2024},
    doi = {10.1101/2024.05.23.595630},
    url = {https://www.biorxiv.org/content/10.1101/2024.05.23.595630v3},
    publisher = {Cold Spring Harbor Laboratory},
    note = {Preprint, version 3}
}
```

## 📄 License

MIT License - see `LICENSE` file for details.
