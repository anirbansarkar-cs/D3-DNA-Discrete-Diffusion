#!/bin/bash
# Thin wrapper that prints the d3-new install workflow.
# The real work is split between environment_new.yml (conda) and
# post_install.sh (pip steps that yml cannot express).

set -e

ENV_NAME="d3-new"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
fi

cat <<EOF
==============================================
D3-DNA-Discrete-Diffusion environment setup
==============================================

This script will run, in order:
  1) ${CONDA_CMD} env create -f environment_new.yml -n ${ENV_NAME}
  2) ${CONDA_CMD} activate ${ENV_NAME}
  3) bash post_install.sh

If you'd rather run these steps yourself, abort now (Ctrl-C) and run
them manually from the repo root.

==============================================
EOF

eval "$(${CONDA_CMD} shell.bash hook)"

if ${CONDA_CMD} env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists - skipping create."
else
    echo "[1/3] Creating '${ENV_NAME}' from environment_new.yml..."
    ${CONDA_CMD} env create -f "${HERE}/environment_new.yml" -n "${ENV_NAME}"
fi

echo "[2/3] Activating '${ENV_NAME}'..."
${CONDA_CMD} activate "${ENV_NAME}"

echo "[3/3] Running post_install.sh..."
bash "${HERE}/post_install.sh"

echo ""
echo "Setup complete. Activate later with: ${CONDA_CMD} activate ${ENV_NAME}"
