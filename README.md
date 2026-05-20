# D3-DNA-Discrete-Diffusion

This repo contains a PyTorch implementation for the paper "Designing DNA With Tunable Regulatory Activity Using Discrete Diffusion". The training and sampling part of the code is inspired by [Score entropy discrete diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).


## 📁 Project Structure

```
D3-DNA-Discrete-Diffusion/
├── scripts/                    # Base classes with shared functionality
│   ├── train.py               # Base training (absorbs Lightning functionality)
│   ├── evaluate.py            # Base evaluation with common metrics
│   └── sample.py              # Base sampling with DDPM/DDIM algorithms
├── model_zoo/                 # Dataset-specific implementations
│   ├── deepstarr/             # DeepSTARR dataset (249bp enhancers)
│   │   ├── configs/           # Dataset-specific configurations
│   │   ├── data.py           # Data loading and preprocessing
│   │   ├── models.py         # Model architectures
│   │   ├── train.py          # Inherits from scripts.train
│   │   ├── evaluate.py       # Inherits from scripts.evaluate
│   │   ├── sample.py         # Inherits from scripts.sample
│   │   └── sp_mse_callback.py # Dataset-specific SP-MSE validation
│   ├── mpra/                 # MPRA dataset (200bp regulatory)
│   └── promoter/             # Promoter dataset (1024bp with expression)
├── utils/                    # Shared utilities
│   ├── data_utils.py         # Common data processing (cycle_loader, etc.)
│   ├── sp_mse_callback.py    # Base SP-MSE callback class
│   └── [other shared utilities...]
├── model/                    # Core model components
└── pyproject.toml           # Package configuration
```

## ⚡ Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/your-repo/D3-DNA-Discrete-Diffusion.git
cd D3-DNA-Discrete-Diffusion

# Install in development mode (recommended)
pip install -e .
```

### Training Models
```bash
# Train DeepSTARR with transformer
python model_zoo/deepstarr/train.py --architecture transformer

# Train MPRA with convolutional architecture  
python model_zoo/mpra/train.py --architecture convolutional

# Train Promoter with custom config
python model_zoo/promoter/train.py --architecture transformer --config custom.yaml
```

### Evaluation
```bash
# Evaluate model performance
python model_zoo/deepstarr/evaluate.py --architecture transformer --checkpoint model.ckpt

# Evaluate with oracle model (SP-MSE)
python model_zoo/deepstarr/evaluate.py --architecture transformer --checkpoint model.ckpt --use_oracle --oracle_checkpoint oracle.ckpt
```

### Sampling
```bash
# Generate sequences
python model_zoo/deepstarr/sample.py --architecture transformer --checkpoint model.ckpt --num_samples 1000

# Generate with specific targets (DeepSTARR)
python model_zoo/deepstarr/sample.py --architecture transformer --checkpoint model.ckpt --dev_activity 2.0 --hk_activity 1.5

# Generate promoters with expression targets
python model_zoo/promoter/sample.py --architecture transformer --checkpoint model.ckpt --expression_target 3.0
```

## 🧬 Supported Datasets

### DeepSTARR
- **Purpose**: Enhancer activity prediction
- **Sequence Length**: 249 bp
- **Labels**: 2 (developmental + housekeeping enhancer activities)
- **Oracle**: PL_DeepSTARR model for SP-MSE evaluation

### MPRA (Massively Parallel Reporter Assay)
- **Purpose**: Regulatory sequence analysis
- **Sequence Length**: 200 bp  
- **Labels**: 3 (regulatory activity measurements)
- **Oracle**: PL_mpra model for SP-MSE evaluation

### Promoter
- **Purpose**: Gene expression prediction
- **Sequence Length**: 1024 bp
- **Labels**: Expression values (concatenated with sequences)
- **Oracle**: SEI (Sequence-to-Expression and Interaction) model

## 🔧 Model Architectures

### Transformer
- Multi-head attention with positional embeddings
- Layer normalization and residual connections
- Configurable depth (n_blocks) and width (hidden_size)
- Conditional generation with label embeddings

### Convolutional
- Multi-scale convolutional layers
- Residual connections and batch normalization
- Adaptive pooling for variable-length inputs
- Efficient for longer sequences

## 📊 Advanced Features

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

### Custom Sampling Methods
- **DDPM**: Standard denoising diffusion
- **DDIM**: Faster deterministic sampling
- **Conditional Generation**: Target-specific sequence generation

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
- **MPRA**: Accurate regulatory predictions  
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
