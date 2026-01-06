---
name: sampler

description: Launch and manage D3 (DNA Discrete Diffusion) sampling experiments on SLURM clusters. This skill handles job submission, WandB tracking, and result management for genomic datasets (LentiMPRA, DeepSTARR, Promoter).
---

## When to Use This Skill
Use when users need to:
- Generate sequences from trained D3 diffusion models
- Run sampling across genomic datasets
- Explore sampling hyperparameters (timesteps, conditioning, initial conditions)
- Track results with WandB
- Launch parallel sampling jobs
- Perform conditional generation with specific activities
- Run inpainting experiments with motif constraints

## Critical Directives

1. **ALWAYS submit via SLURM** - Use `sbatch created_script.sh`, NEVER direct python execution
2. **Store scripts with datetime naming** - Save to `/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/slurm_scripts/YYYYMMDD_HHMMSS.sh`
3. **Use array jobs for parameter sweeps** - For SAME dataset with different parameters, create array jobs. For DIFFERENT datasets, create separate scripts.
4. **Include WandB logging** - always enable with `--use_wandb` and provide descriptive project/run names unless explicitly indicated to NOT use wandb
5. **Verify checkpoints exist** - Confirm checkpoint paths before submission
6. **Specify architecture explicitly** - Always include `--architecture transformer` or `--architecture convolutional`

## Prerequisites Checklist
**BEFORE any job launch, verify ALL items:**

### ✅ Environment Setup
- SLURM cluster access configured
- Conda environment `d3-old` exists with all dependencies
- WandB authenticated (`wandb login`) if using `--use_wandb`

### ✅ Required Paths Exist
- Checkpoint file exists at specified path
- Data file exists if using `--data_path` (required for test set conditioning/representation saving)
- Custom initialization files exist if using `--custom_inits_path`
- Inpainting CSV files exist if using `--inpainting_csv`
- Output directory is writable

### ✅ WandB Configuration (if using)
- `--use_wandb` flag enabled
- Project name meaningful (defaults to `{dataset}-sampling`)
- Run name clearly identifies sampling configuration
- Entity/team set with `--wandb_entity` if using team account

### ⚠️ Critical Settings
- **SLURM time limit exceeds sampling time** - Default limits may be too short
- **GPU resources match model requirements** - Verify GPU memory suffices for model + batch size
- **Checkpoint compatibility** - Ensure checkpoint matches dataset and model architecture
- **Architecture specification** - Always provide `--architecture` argument (transformer or convolutional per user indication)

## Dataset and Model Configurations

### DeepSTARR Dataset
**Purpose:** Drosophila enhancer activity prediction
```bash
CHECKPOINT=/grid/koo/home/shared/d3/trained_weights/deepstarr/d3-tran/model-epoch=279-val_loss=319.2632.ckpt
DATA=/grid/koo/home/shared/d3/data/deepstarr/DeepSTARR_data_with_dinuc.h5
ORACLE=/grid/koo/home/shared/d3/oracle_weights/deepstarr/oracle_DeepSTARR_DeepSTARR_data.ckpt
SCRIPT=/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/sample.py
```

### Promoter Dataset
**Purpose:** Human promoter sequence generation
```bash
CHECKPOINT=/grid/koo/home/shared/d3/trained_weights/promoter_09242025/model-epoch=175-val_loss=1119.9065.ckpt
DATA=/grid/koo/home/shared/d3/data/promoter/Promoter_data.npz
ORACLE=/grid/koo/home/shared/d3/oracle_weights/promoter/best.sei.model.pth.tar
SCRIPT=/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/promoter/sample.py
```

### LentiMPRA HepG2
**Purpose:** Massively parallel reporter assay for hepg2
```bash
CHECKPOINT=/grid/koo/home/shared/d3/trained_weights/lentimpra/d3-tran-Hepg2/model-epoch=259-val_loss=250.9961.ckpt
DATA=/grid/koo/home/shared/d3/data/lentimpra/lenti_MPRA_HepG2_data_with_dinuc.h5
ORACLE=/grid/koo/home/shared/d3/oracle_weights/lentimpra/HepG2/best_model-epoch=15-val_pearson=0.741.ckpt
SCRIPT=/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/sample.py
```

### LentiMPRA K562
**Purpose:** Massively parallel reporter assay for k562
```bash
CHECKPOINT=/grid/koo/home/shared/d3/trained_weights/lentimpra/d3-tran-K562/model-epoch=263-val_loss=247.4976.ckpt
DATA=/grid/koo/home/shared/d3/data/lentimpra/lenti_MPRA_K562_data_with_dinuc.h5
ORACLE=/grid/koo/home/shared/d3/oracle_weights/lentimpra/best_model-epoch=24-val_pearson=0.814.ckpt
SCRIPT=/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/sample.py
```

### LentiMPRA WTC11
**Purpose:** Massively parallel reporter assay for wtc11
```bash
DATA=/grid/koo/home/shared/d3/data/lentimpra/lenti_MPRA_WTC11
ORACLE=/grid/koo/home/shared/d3/oracle_weights/lentimpra/WTC11/best_model_wtc11-epoch=28-val_pearson=0.712.ckpt
SCRIPT=/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/sample.py
```
**⚠️ WARNING:** Checkpoint not available for WTC11. Stop and warn user if asked to use this dataset.

### LentiMPRA 60k Dataset
**Purpose:** Massively parallel reporter assay across 3 cell lines (K562, HepG2, WTC11)
```bash
CHECKPOINT=/grid/koo/home/shared/d3/trained_weights/lentimpra_60k/model-epoch=299-val_loss=220.3063.ckpt
DATA=/grid/koo/home/shared/d3/data/lentimpra/60k/dataset_3cells_230bp_cleaned.h5
SCRIPT=/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/sample.py
```
**Notes:**
- Contains ~60,000 total samples
- Multi-cell line model supports activity conditioning for K562, HepG2, and WTC11
- Specify activities per cell line or use random/test set conditioning
- Use `--architecture transformer_multi_class`
- Oracles use previous LentiMPRA model oracles

## Essential Command Line Arguments

### REQUIRED Arguments (Always Include)
**`--checkpoint`** - Path to trained model checkpoint
**`--architecture`** - Model architecture type (`transformer` or `convolutional`)

### Core Sampling Arguments
**`--num_samples`** (default: 1000) - Sequences to generate
**`--steps`** (optional) - Sampling/denoising steps (defaults to sequence length)
**`--batch_size`** (default: 256) - Reduce if GPU OOM occurs
**`--start_at_timestep`** (default: 0) - Start sampling at this timestep (0 = pure noise)

### Output Configuration
**`--output`** (optional) - Output file path (auto-generated if not specified)
**`--format`** (default: h5) - Output format: `npz`, `fasta`, `csv`, `h5`, `hdf5`, `pt`
**`--sequence_encoding`** (default: index) - Encoding: `index` (0-3) or `onehot` (one-hot vectors)
**`--save_elements`** (optional) - Elements to save: `sequence`, `score`, `activity_label`

### Initial Conditions
**`--initial_condition`** (default: random) - Choices: `random`, `test`, `dinuc`, `custom`
**`--data_path`** (REQUIRED when using: `--initial_condition test/dinuc`, `--use_test_set`, `--save_rep`)

### Conditioning
**`--use_test_set`** (flag) - Use test set labels from dataset as conditioning (requires `--data_path`)

### LentiMPRA-Specific
**Single-class activity:** `--activity` (float)
**Multi-class activities:** `--k562_activity`, `--hepg2_activity`, `--wtc11_activity` (floats)
**Unconditional sampling:** `--unconditional` (flag)
**Custom initialization:** `--custom_inits_path` (H5 file path)
**Inpainting:** `--inpainting_csv` + `--pattern_csv` (both required)

### WandB Logging
**`--use_wandb`** (flag) - Enable WandB logging
**`--wandb_project`** (optional) - Project name (defaults to `{dataset}-sampling`)
**`--wandb_name`** (optional) - Run name (auto-generated if not specified)
**`--wandb_entity`** (optional) - Entity/team name
**`--wandb_tags`** (optional) - Run tags

## SLURM Script Templates

### Standard Job Script
```bash
#!/bin/bash
#SBATCH --job-name=d3_sample_<descriptive_name>
#SBATCH --output=/grid/koo/home/aduran/logs/%j.out
#SBATCH --error=/grid/koo/home/aduran/logs/%j.err
#SBATCH --time=HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

source ~/.bashrc
mamba activate d3-old

python /path/to/sample.py \
    --checkpoint <checkpoint_path> \
    --architecture transformer \
    --num_samples 1000 \
    [additional arguments]
```

### Array Job Template (Parameter Sweeps)
```bash
#!/bin/bash
#SBATCH --job-name=d3_sweep_<dataset>
#SBATCH --array=0-11  # Adjust based on configurations
#SBATCH --output=/grid/koo/home/aduran/logs/%A_%a.out
#SBATCH --error=/grid/koo/home/aduran/logs/%A_%a.err
#SBATCH --time=HH:MM:SS
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
mamba activate d3-old

# Define parameter arrays
STEPS=(50 100 200 500)
BATCH_SIZES=(128 256 512)

# Calculate indices
STEP_IDX=$((SLURM_ARRAY_TASK_ID / ${#BATCH_SIZES[@]}))
BATCH_IDX=$((SLURM_ARRAY_TASK_ID % ${#BATCH_SIZES[@]}))

STEP=${STEPS[$STEP_IDX]}
BATCH=${BATCH_SIZES[$BATCH_IDX]}

RUN_NAME="sweep_s${STEP}_b${BATCH}"

python /path/to/sample.py \
    --checkpoint <checkpoint_path> \
    --architecture transformer \
    --num_samples 1000 \
    --steps $STEP \
    --batch_size $BATCH \
    --use_wandb \
    --wandb_project "d3-sampling" \
    --wandb_name "$RUN_NAME"
```

## Example Scripts
Reference examples in: `/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/skills/sampling/scripts`

These show:
- Correct SLURM directives
- Proper environment activation
- Standard argument patterns per dataset
- WandB integration
- Array job indexing
- Conditional generation
- Inpainting workflows

## WandB Best Practices

### Project Naming
Format: `d3-{dataset}-{experiment_type}`

**Examples:**
- `d3-deepstarr-sampling`
- `d3-lentimpra-conditional`
- `d3-promoter-steps-sweep`

### Run Naming
Format: `<descriptor>_<param1>_<param2>_<param3>`

**Examples:**
- `baseline_s100_n5000` (baseline, 100 steps, 5000 samples)
- `conditional_k562high_s200` (conditional K562 high activity, 200 steps)
- `delayed_t100_dinuc` (delayed from t=100, dinuc init)

### Test Runs (5-10 minutes)
When user asks for 'test run':
- Small sample count: 500-1000
- Few steps: 10
- Single configuration
- Purpose: Verify setup before larger experiments

## Common Failure Modes & Fixes

### Out of GPU Memory (OOM)
**Fix sequence:**
1. Reduce `--batch_size` (256 → 128 → 64)
2. Reduce `--num_samples` per job, run multiple jobs
3. Disable trajectory saving: omit `--save_elements score`
4. Request more GPU memory: `#SBATCH --gres=gpu:a100:1`
5. Verify architecture matches checkpoint

### Missing Required Arguments
**Error:** `error: the following arguments are required: --checkpoint, --architecture`
**Fix:** ALWAYS include `--checkpoint` and `--architecture`

### Data Path Required But Missing
**Error:** Occurs when using `--use_test_set`, `--initial_condition test/dinuc`, or `--save_rep`
**Fix:** Add `--data_path` with correct dataset path from configurations above

### SLURM Job Timeout
**Fix:**
1. Check runtime: `sacct -j <job_id> --format=JobID,Elapsed`
2. Increase `#SBATCH --time=` with 30% buffer
3. Reduce samples per job (smaller `--num_samples`)
4. Reduce steps (`--steps 100` instead of `--steps 500`)

### Inpainting CSV Format Error
**Fix:**
1. Verify CSV file format matches expected structure
2. Ensure BOTH `--pattern_csv` and `--inpainting_csv` paths provided
3. Check pattern names match between CSV files
4. Validate CSV readability: `head -n 5 /path/to/file.csv`

## Script Storage

### Directory Structure
```
/Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/
├── slurm_scripts/           # Generated scripts (datetime named)
│   ├── 20260105_143022.sh
│   └── ...
└── skills/
    └── scripts/             # Example/reference scripts
```

### Naming Convention
**Format:** `YYYYMMDD_HHMMSS.sh`
- Ensures chronological ordering
- Prevents filename conflicts
- Easy to identify submission time

**Directory Creation:**
```bash
mkdir -p /Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/slurm_scripts/
```

## Job Submission Workflow

### 5-Step Process
1. **Verify ALL prerequisites** (use checklist above)
2. **Create SLURM script** with datetime filename
3. **Review script** with user, focusing on:
   - Checkpoint and architecture match
   - Required arguments present (`--checkpoint`, `--architecture`)
   - Data path included if needed
   - WandB configuration if tracking desired
4. **Submit job:**
   ```bash
   sbatch /Users/alejandraduran/Documents/D3-DNA-Discrete-Diffusion/slurm_scripts/YYYYMMDD_HHMMSS.sh
   ```
5. **Report submission details**

### After Submission - Provide User With:
- ✅ **Job ID** and status command: `squeue -u $USER`
- ✅ **Log file paths**: stdout and stderr locations
- ✅ **WandB project URL** (if `--use_wandb` enabled)
- ✅ **Expected completion time**
- ✅ **Output file location**