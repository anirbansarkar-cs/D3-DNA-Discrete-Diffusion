#!/bin/bash
#SBATCH --job-name=rndN_dps
#SBATCH --output=sbatch_out/rounded_pipeline/roundN_%A_%a_stdout.out
#SBATCH --error=sbatch_out/rounded_pipeline/roundN_%A_%a_stderr.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=asarkar@cshl.edu

# ===========================================================================
# Rounded Pipeline — Round N DPS Push (Template for subsequent rounds)
# ===========================================================================
# Parameterized by ROUND_NUM. Seeds come from collect_round.py output.
#
# Suggested activity progression (conservative ramp):
#   Round 1: act=3,4,5,6,7     (starting from real data, mean ~1.47)
#   Round 2: act=4,5,6,7,8     (after seeds improve to ~2-3 mean)
#   Round 3: act=5,6,7,8,9     (after seeds improve to ~3-4 mean)
#   Round 4: act=6,7,8,9,10    (after seeds improve to ~5+ mean)
#
# Suggested grid narrowing based on round 1 results:
#   Round 1: broad (8w × 5eta × 3nf = 120 conditions)
#   Round 2: narrowed based on R1 results (best w, eta, nf ranges)
#   Round 3+: focused on sweet spot
#
# Usage (override --output/--error so log filenames include the round number):
#   ROUND_NUM=2 sbatch --array=0-4 \
#     --output=sbatch_out/rounded_pipeline/round2_%A_%a_stdout.out \
#     --error=sbatch_out/rounded_pipeline/round2_%A_%a_stderr.out \
#     scripts/rounded_runs/run_roundN_dps_push.sh
#
# Smoke test:
#   ROUND_NUM=2 NUM_SAMPLES=100 sbatch --array=0 scripts/rounded_runs/run_roundN_dps_push.sh
# ===========================================================================

# ---- Environment ----------------------------------------------------------
source /grid/it/data/elzar/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate d3_cuda118

set -euo pipefail

# ---- Round parameters (REQUIRED) -----------------------------------------
ROUND_NUM="${ROUND_NUM:?ERROR: ROUND_NUM must be set (e.g. ROUND_NUM=2)}"
PREV_ROUND=$((ROUND_NUM - 1))

# ---- Paths ----------------------------------------------------------------
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/rounded_pipeline

CKPT="${CKPT:-/grid/koo/home/asarkar/CFG-SDDD/experiments/lentimpra/20260206_225457/checkpoints/sp-mse_0.184464_step_822870.ckpt}"
ORACLE_CKPT="${ORACLE_CKPT:-/grid/koo/home/shared/d3/oracle_weights/lentimpra/best_model-epoch=24-val_pearson=0.814.ckpt}"
WARM_START_PATH="${WARM_START_PATH:-results/rounded_pipeline/seed_pool_round${ROUND_NUM}/pool.h5}"

# ---- Array task mapping ---------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Activity progression — adjust these per round based on seed pool oracle scores
# Default: +1 per round offset from base [3,4,5,6,7]
BASE_ACT_OFFSET=$((ROUND_NUM - 1))
ACTIVITY_VALUES_STR="${ACTIVITY_VALUES_STR:-}"
if [[ -z "$ACTIVITY_VALUES_STR" ]]; then
    # Default progression: shift up by round number - 1
    A0=$(echo "3.0 + $BASE_ACT_OFFSET" | bc)
    A1=$(echo "4.0 + $BASE_ACT_OFFSET" | bc)
    A2=$(echo "5.0 + $BASE_ACT_OFFSET" | bc)
    A3=$(echo "6.0 + $BASE_ACT_OFFSET" | bc)
    A4=$(echo "7.0 + $BASE_ACT_OFFSET" | bc)
    ACTIVITY_VALUES=($A0 $A1 $A2 $A3 $A4)
else
    read -ra ACTIVITY_VALUES <<< "$ACTIVITY_VALUES_STR"
fi
ACTIVITY=${ACTIVITY_VALUES[$TASK_ID]}

GC_GRAD_WEIGHT=0

# ---- Inner sweep grid (narrow after round 1) ----------------------------
# Default: same broad grid; override with environment variables after analyzing
# previous round results
GUIDANCE_WEIGHTS="${GUIDANCE_WEIGHTS:-0.5 1 2 5 8 10 15 20}"
ETAS="${ETAS:-500 1000 2000 3000 5000}"
NOISE_FRACTIONS="${NOISE_FRACTIONS:-0.01 0.05 0.1}"

# ---- Sampling -------------------------------------------------------------
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
STEPS="${STEPS:-20}"
SAMPLING_BATCH_SIZE="${SAMPLING_BATCH_SIZE:-512}"

# ---- GC targets -----------------------------------------------------------
GC_TARGET_LOW="${GC_TARGET_LOW:-0.40}"
GC_TARGET_HIGH="${GC_TARGET_HIGH:-0.60}"

# ---- DPS params -----------------------------------------------------------
TAU_START="${TAU_START:-1.0}"
TAU_END="${TAU_END:-0.1}"
GC_GRAD_TARGET="${GC_GRAD_TARGET:-0.50}"
ETA_SCHEDULE="${ETA_SCHEDULE:-sqrt}"

# ---- Archives (post-hoc filter) ------------------------------------------
ORACLE_THRESHOLD="${ORACLE_THRESHOLD:-1.0}"
GC_THRESHOLD="${GC_THRESHOLD:-0.55}"
GC_THRESHOLD_LOW="${GC_THRESHOLD_LOW:-0.45}"

# ---- Output directory -----------------------------------------------------
OUTPUT_DIR="results/rounded_pipeline/round${ROUND_NUM}_dps_push/act${ACTIVITY}"

# ---- Config ---------------------------------------------------------------
CONFIG="${CONFIG:-}"

# ===========================================================================
# Build command
# ===========================================================================
CMD=(
    python model_zoo/lentimpra/cfg_sweep.py
    --checkpoint "$CKPT"
    --oracle_checkpoint "$ORACLE_CKPT"
    --mode warmstart_dps
    --warm_start_path "$WARM_START_PATH"
    --guidance_weights $GUIDANCE_WEIGHTS
    --gc_penalties 0.0
    --gc_target_low "$GC_TARGET_LOW"
    --gc_target_high "$GC_TARGET_HIGH"
    --etas $ETAS
    --noise_fractions $NOISE_FRACTIONS
    --activities "$ACTIVITY"
    --cfg_method logit
    --gc_method fixed
    --anneal_strategy none
    --gc_penalty_start_frac 0.0
    --tau_start "$TAU_START"
    --tau_end "$TAU_END"
    --guide_start_frac 0.0
    --gc_grad_weight "$GC_GRAD_WEIGHT"
    --gc_grad_target "$GC_GRAD_TARGET"
    --eta_schedule "$ETA_SCHEDULE"
    --num_samples "$NUM_SAMPLES"
    --steps "$STEPS"
    --sampling_batch_size "$SAMPLING_BATCH_SIZE"
    --output_dir "$OUTPUT_DIR"
    --oracle_threshold "$ORACLE_THRESHOLD"
    --gc_threshold "$GC_THRESHOLD"
    --gc_threshold_low "$GC_THRESHOLD_LOW"
)

if [[ -n "$CONFIG" ]]; then
    CMD+=(--config "$CONFIG")
fi

# ===========================================================================
# Run
# ===========================================================================
echo "========================================================================"
echo "Rounded Pipeline — Round ${ROUND_NUM} DPS Push"
echo "========================================================================"
echo "ROUND_NUM:         $ROUND_NUM"
echo "PREV_ROUND:        $PREV_ROUND"
echo "TASK_ID:           $TASK_ID"
echo "Activity:          $ACTIVITY"
echo "Activities avail:  ${ACTIVITY_VALUES[*]}"
echo "GC grad weight:    $GC_GRAD_WEIGHT (fixed)"
echo "Warm-start path:   $WARM_START_PATH"
echo "Guidance weights:  $GUIDANCE_WEIGHTS"
echo "Etas:              $ETAS"
echo "Noise fractions:   $NOISE_FRACTIONS"
echo "Tau:               $TAU_START -> $TAU_END"
echo "Eta schedule:      $ETA_SCHEDULE"
echo "GC target:         [$GC_TARGET_LOW, $GC_TARGET_HIGH]"
echo "Post-hoc filter:   [$GC_THRESHOLD_LOW, $GC_THRESHOLD]"
echo "Num samples:       $NUM_SAMPLES"
echo "Output dir:        $OUTPUT_DIR"
echo "Command: ${CMD[*]}"
echo "========================================================================"

"${CMD[@]}"
