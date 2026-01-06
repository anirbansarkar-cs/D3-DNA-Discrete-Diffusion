### Example 5: Inpainting with Motif Constraints (LentiMPRA)

**User Request:** "Generate sequences with specific motif patterns constrained via inpainting"

**Script:** `/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/slurm_scripts/20260105_150000.sh`

```bash
#!/bin/bash
#SBATCH --job-name=d3_lentimpra_inpainting
#SBATCH --output=/grid/koo/home/aduran/logs/%j.out
#SBATCH --error=/grid/koo/home/aduran/logs/%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G

source ~/.bashrc
mamba activate d3-old

python /Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/sample.py \
    --checkpoint /grid/koo/home/shared/d3/trained_weights/lentimpra_60k/model-epoch=299-val_loss=220.3063.ckpt \
    --architecture transformer \
    --num_samples 2000 \
    --steps 200 \
    --batch_size 128 \
    --inpainting_csv /path/to/Dev_high_hits.csv \
    --pattern_csv /path/to/patterns.csv \
    --k562_activity 2.5 \
    --save_elements sequence activity_label \
    --use_wandb \
    --wandb_project "d3-lentimpra-inpainting" \
    --wandb_name "dev_motif_constrained"
```
