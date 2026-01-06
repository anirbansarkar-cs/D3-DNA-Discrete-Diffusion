### Example 4: Parameter Sweep Array Job (Sampling Steps)

**User Request:** "Sweep sampling steps [50, 100, 200, 500, 1000] for DeepSTARR"

**Script:** `/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/slurm_scripts/20260105_145030.sh`

```bash
#!/bin/bash
#SBATCH --job-name=d3_deepstarr_steps_sweep
#SBATCH --array=0-4  # 5 different step values
#SBATCH --output=/grid/koo/home/aduran/logs/%A_%a.out
#SBATCH --error=/grid/koo/home/aduran/logs/%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
mamba activate d3-old

# Parameter array
STEPS=(50 100 200 500 1000)

# Get step value for this array task
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}

# Create unique run name
RUN_NAME="steps_sweep_s${STEP}"

python /Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/sample.py \
    --checkpoint /grid/koo/home/shared/d3/trained_weights/deepstarr/d3-tran/model-epoch=279-val_loss=319.2632.ckpt \
    --architecture transformer \
    --num_samples 1000 \
    --steps $STEP \
    --batch_size 256 \
    --use_wandb \
    --wandb_project "d3-deepstarr-steps-sweep" \
    --wandb_name "$RUN_NAME" \
    --wandb_tags sweep steps-comparison
```
