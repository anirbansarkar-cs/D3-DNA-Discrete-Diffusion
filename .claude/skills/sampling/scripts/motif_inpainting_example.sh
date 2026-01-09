### Example 5: Inpainting with Motif Constraints (DeepSTARR)

**User Request:** "Generate sequences with motif regions constrained via inpainting"

**Script:** `/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/slurm_scripts/20260108_inpainting.sh`

**Mode 1: Inpaint Motifs (Fix outside, generate inside motif regions)**
```bash
#!/bin/bash
#SBATCH --job-name=d3_deepstarr_inpaint_motifs
#SBATCH --output=/grid/koo/home/aduran/logs/%j.out
#SBATCH --error=/grid/koo/home/aduran/logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
mamba activate d3-old

# Inpaint motifs: fix positions OUTSIDE motif regions, generate INSIDE
python /Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/sample.py \
    --checkpoint /grid/koo/home/shared/d3/trained_weights/deepstarr/d3-tran/model-epoch=279-val_loss=319.2632.ckpt \
    --architecture transformer \
    --inpainting_mode inpaint_motifs \
    --inpainting_data /path/to/inpainting_data/all_hits_combined.h5 \
    --inpainting_seed 42 \
    --steps 249 \
    --batch_size 256 \
    --output /grid/koo/home/aduran/results/deepstarr_inpaint_motifs \
    --format h5 \
    --save_elements sequence score \
    --use_wandb \
    --wandb_project "d3-deepstarr-inpainting" \
    --wandb_name "inpaint_motifs_seed42"
```

**Mode 2: Inpaint Not Motifs (Fix inside, generate outside motif regions)**
```bash
#!/bin/bash
#SBATCH --job-name=d3_deepstarr_inpaint_not_motifs
#SBATCH --output=/grid/koo/home/aduran/logs/%j.out
#SBATCH --error=/grid/koo/home/aduran/logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
mamba activate d3-old

# Inpaint NOT motifs: fix positions INSIDE motif regions, generate OUTSIDE
python /Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/sample.py \
    --checkpoint /grid/koo/home/shared/d3/trained_weights/deepstarr/d3-tran/model-epoch=279-val_loss=319.2632.ckpt \
    --architecture transformer \
    --inpainting_mode inpaint_not_motifs \
    --inpainting_data /path/to/inpainting_data/all_hits_combined.h5 \
    --inpainting_seed 42 \
    --steps 249 \
    --batch_size 256 \
    --output /grid/koo/home/aduran/results/deepstarr_inpaint_not_motifs \
    --format h5 \
    --use_wandb \
    --wandb_project "d3-deepstarr-inpainting" \
    --wandb_name "inpaint_not_motifs_seed42"
```

**Notes:**
- Inpainting data file (`all_hits_combined.h5`) must contain:
  - `X`: One-hot encoded sequences (658, 249, 4)
  - `Y_target`: Activity values (658, 2) - used for conditioning
  - Position data: `start_dev`, `end_dev`, `start_hk`, `end_hk`
- When inpainting mode is active, `--num_samples` is ignored (generates all 658 samples)
- Y_target values from h5 file are used for conditioning (ignores `--dev_activity`/`--hk_activity`)
- `--inpainting_seed` controls random selection when both dev/hk positions available
- For sequences with NaN positions, no constraints are applied
