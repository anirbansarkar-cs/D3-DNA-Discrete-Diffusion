### Example 2: Conditional Sampling with Test Set Labels (Promoter)

**User Request:** "Generate sequences using test set conditioning from the promoter dataset"

**Script:** `/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/slurm_scripts/20260105_143525.sh`

```bash
#!/bin/bash
#SBATCH --job-name=d3_promoter_conditional
#SBATCH --output=/grid/koo/home/aduran/logs/%j.out
#SBATCH --error=/grid/koo/home/aduran/logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
mamba activate d3-old

python /Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/promoter/sample.py \
    --checkpoint /grid/koo/home/shared/d3/trained_weights/promoter_09242025/model-epoch=175-val_loss=1119.9065.ckpt \
    --architecture transformer \
    --data_path /grid/koo/home/shared/d3/data/promoter/Promoter_data.npz \
    --num_samples 2000 \
    --steps 200 \
    --use_test_set \
    --initial_condition test \
    --save_elements sequence activity_label \
    --use_wandb \
    --wandb_project "d3-promoter-sampling" \
    --wandb_name "conditional_test_s200"
