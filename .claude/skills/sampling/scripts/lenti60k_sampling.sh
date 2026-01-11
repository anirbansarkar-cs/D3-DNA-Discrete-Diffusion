### Example 3: Multi-Activity Conditional Sampling (LentiMPRA)

**User Request:** "Generate sequences with high activity in K562 (3.0), medium in HepG2 (2.0), low in WTC11 (1.0)"

**Script:** `/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/slurm_scripts/20260105_144012.sh`

```bash
#!/bin/bash
#SBATCH --job-name=d3_lentimpra_multiactivity
#SBATCH --partition=gpuq
#SBATCH --output=/grid/koo/home/aduran/logs/%j.out
#SBATCH --error=/grid/koo/home/aduran/logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G

source ~/.bashrc
mamba activate d3-new

python /Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/sample.py \
    --checkpoint /grid/koo/home/shared/d3/trained_weights/lentimpra_60k/model-epoch=299-val_loss=220.3063.ckpt \
    --architecture transformer \
    --num_samples 5000 \
    --steps 100 \
    --batch_size 256 \
    --k562_activity 3.0 \
    --hepg2_activity 2.0 \
    --wtc11_activity 1.0 \
    --save_elements sequence activity_label \
    --format h5 \
    --use_wandb \
    --wandb_project "d3-lentimpra-conditional" \
    --wandb_name "k562high_hepg2med_wtc11low"
