#!/bin/bash
# =============================================================================
# D3-DNA-Discrete-Diffusion Environment Setup Script
# =============================================================================
# This script creates the 'd3-new' environment with PyTorch 2.5.1 + CUDA 12.1
# and flash-attn 2.5.0 using a 2-step install approach.
#
# The 2-step install trick:
# 1. First, install the base environment WITHOUT flash-attn
# 2. Then, install flash-attn separately with --no-build-isolation
#
# This avoids build conflicts where flash-attn tries to compile against
# a different CUDA version than what PyTorch was built with.
# =============================================================================

set -e  # Exit on error

ENV_NAME="d3-new"
PYTHON_VERSION="3.9"
CUDA_VERSION="11.8.0"

echo "=============================================="
echo "D3-DNA Environment Setup"
echo "=============================================="

# =============================================================================
# Step 0: Load required modules
# =============================================================================
echo "[Step 0] Loading CUDA module..."
module load cuda11.8/toolkit/${CUDA_VERSION} 2>/dev/null || echo "Module load skipped (may not be on HPC)"

# Initialize mamba/conda
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
fi
echo "Using: ${CONDA_CMD}"

# Source conda/mamba
eval "$(${CONDA_CMD} shell.bash hook)"

# =============================================================================
# Step 1: Create or activate the environment
# =============================================================================
echo ""
echo "[Step 1] Setting up conda environment '${ENV_NAME}'..."

if ${CONDA_CMD} env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' exists. Activating..."
    ${CONDA_CMD} activate ${ENV_NAME}
else
    echo "Creating new environment '${ENV_NAME}'..."
    ${CONDA_CMD} create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    ${CONDA_CMD} activate ${ENV_NAME}
fi

# =============================================================================
# Step 2: Install CUDA toolkit from nvidia channel
# =============================================================================
echo ""
echo "[Step 2] Installing CUDA toolkit from conda..."
${CONDA_CMD} install -c nvidia/label/cuda-${CUDA_VERSION} cuda-toolkit -y

# =============================================================================
# Step 3: Install PyTorch with CUDA 12.1 (2-step trick - Part 1)
# =============================================================================
echo ""
echo "[Step 3] Installing PyTorch 2.5.1 with CUDA 12.1..."
echo "         (flash-attn will be installed separately)"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# =============================================================================
# Step 4: Install core ML dependencies (before flash-attn)
# =============================================================================
echo ""
echo "[Step 4] Installing core ML dependencies..."
pip install \
    pytorch-lightning==2.1.0 \
    transformers==4.38.1 \
    accelerate==0.27.2 \
    einops==0.7.0 \
    wandb==0.16.3 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    hydra-submitit-launcher==1.2.0 \
    dm-tree==0.1.8

# =============================================================================
# Step 5: Install data science packages
# =============================================================================
echo ""
echo "[Step 5] Installing data science packages..."
pip install \
    numpy==1.24.1 \
    scipy==1.13.1 \
    pandas==2.2.1 \
    scikit-learn==1.4.0 \
    h5py==3.10.0 \
    datasets==2.17.1

# =============================================================================
# Step 6: Install visualization packages
# =============================================================================
echo ""
echo "[Step 6] Installing visualization packages..."
pip install \
    matplotlib==3.9.4 \
    seaborn==0.13.2 \
    pillow==10.2.0

# Install conda packages that work better from conda
${CONDA_CMD} install -c conda-forge ipykernel ipywidgets biotite -y

# =============================================================================
# Step 7: Install flash-attn (2-step trick - Part 2)
# =============================================================================
echo ""
echo "[Step 7] Installing flash-attn 2.5.0..."
echo "         Using --no-build-isolation to avoid CUDA version conflicts"

# Install build dependencies first
pip install ninja packaging

# Install flash-attn with no build isolation
# This uses the already-installed PyTorch's CUDA headers
pip install flash-attn==2.5.0 --no-build-isolation

# Verify flash-attn installation
python -c "from flash_attn import flash_attn_func; print('flash-attn installed successfully!')"

# =============================================================================
# Step 8: Install remaining packages
# =============================================================================
echo ""
echo "[Step 8] Installing remaining packages..."
pip install \
    transformer-lens==1.14.0 \
    huggingface-hub==0.21.1 \
    tokenizers==0.15.2 \
    safetensors==0.4.2 \
    beartype==0.14.1 \
    jaxtyping==0.2.25 \
    tqdm==4.66.2 \
    rich==13.7.0 \
    pyyaml==6.0.1 \
    protobuf==4.25.3

# =============================================================================
# Step 9: Install the D3 package in development mode
# =============================================================================
echo ""
echo "[Step 9] Installing D3-DNA-Discrete-Diffusion in development mode..."
pip install -e .

# =============================================================================
# Final verification
# =============================================================================
echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "Verifying installation..."
python -c "
import torch
import pytorch_lightning as pl
from flash_attn import flash_attn_func
import transformers
import wandb
import h5py

print('Package versions:')
print(f'  torch: {torch.__version__}')
print(f'  pytorch-lightning: {pl.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda}')
print('')
print('All packages installed successfully!')
"

echo ""
echo "To activate this environment in the future, run:"
echo "  ${CONDA_CMD} activate ${ENV_NAME}"
echo ""
