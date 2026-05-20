#!/bin/bash
# Post-conda-env install steps for D3-DNA-Discrete-Diffusion (env: d3-new).
#
# Prerequisite (run these first):
#   conda env create -f environment_new.yml -n d3-new
#   conda activate d3-new
#
# Then run this script. It installs the things a conda env yml cannot express:
#   - torch / torchvision / torchaudio from the CUDA 12.1 pip index URL
#   - flash-attn from its prebuilt wheel (no source build on a normal Linux box)
#   - editable install of this repository

set -e

echo "[1/4] Installing torch 2.5.1 (cu121)..."
pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[2/4] Installing flash-attn build helpers..."
pip install ninja packaging

echo "[3/4] Installing flash-attn 2.7.4.post1..."
# --no-build-isolation lets pip see the already-installed torch headers if a
# source build is needed. A prebuilt wheel for cu12 + torch2.5 + cp39 ships
# with the v2.7.4.post1 GitHub release, so this should be a wheel download.
pip install flash-attn==2.7.4.post1 --no-build-isolation

echo "[4/4] Installing D3-DNA-Discrete-Diffusion (editable)..."
pip install -e .

echo ""
echo "Verifying install..."
python - <<'PY'
import torch
import flash_attn
print(f"  torch:       {torch.__version__}")
print(f"  cuda avail:  {torch.cuda.is_available()}")
print(f"  cuda ver:    {torch.version.cuda}")
print(f"  flash_attn:  {flash_attn.__version__}")
from flash_attn import flash_attn_func  # noqa: F401
print("  flash_attn_func imported OK")
PY

echo ""
echo "Done. Activate later with: conda activate d3-new"
