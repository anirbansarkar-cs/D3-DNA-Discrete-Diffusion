### Example 1: Basic Unconditional Sampling (DeepSTARR)

**User Request:** "Generate 1000 sequences from the DeepSTARR model with 100 sampling steps"

**Script:** `/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/slurm_scripts/20260105_143022.sh`

```bash
#!/bin/bash
#SBATCH --job-name=d3_deepstarr_sample
#SBATCH --output=/grid/koo/home/aduran/logs/%j.out
#SBATCH --error=/grid/koo/home/aduran/logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
mamba activate d3-new

python /Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/sample.py \
    --checkpoint /grid/koo/home/shared/d3/trained_weights/deepstarr/d3-tran/model-epoch=279-val_loss=319.2632.ckpt \
    --architecture transformer \
    --num_samples 1000 \
    --steps 100 \
    --batch_size 256 \
    --format h5 \
    --use_wandb \
    --wandb_project "d3-deepstarr-sampling" \
    --wandb_name "baseline_s100_n1000"