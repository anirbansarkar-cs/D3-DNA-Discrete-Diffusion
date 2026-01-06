### Example 6: Delayed Sampling (Partial Denoising)

**User Request:** "Start sampling from timestep 100 with dinucleotide-shuffled test sequences"

**Script:** `/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/slurm_scripts/20260105_151500.sh`

```bash
#!/bin/bash
#SBATCH --job-name=d3_promoter_delayed
#SBATCH --output=/grid/koo/home/aduran/logs/%j.out
#SBATCH --error=/grid/koo/home/aduran/logs/%j.err
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
mamba activate d3-old

python /Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/promoter/sample.py \
    --checkpoint /grid/koo/home/shared/d3/trained_weights/promoter_09242025/model-epoch=175-val_loss=1119.9065.ckpt \
    --architecture transformer \
    --data_path /grid/koo/home/shared/d3/data/promoter/Promoter_data.npz \
    --num_samples 1000 \
    --steps 500 \
    --start_at_timestep 100 \
    --initial_condition dinuc \
    --save_elements sequence score \
    --use_wandb \
    --wandb_project "d3-promoter-delayed" \
    --wandb_name "delayed_t100_dinuc"
```
