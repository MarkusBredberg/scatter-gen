#!/bin/bash -l

#SBATCH --job-name=classify_nl0
#SBATCH --account=sk036
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --chdir=/users/mbredber/p2_DCRECLASS
#SBATCH --output=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%A_%a.out
#SBATCH --error=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7

# ── Run configuration ─────────────────────────────────────────────────────────
ALL_CLASSIFIERS=("ImageCNN" "DualCSN" "DualSSN" "SimpleScatterNet")
ALL_VERSIONS=("T25kpc" "RAW")
CROP_MODE="beam_crop"
BLUR_METHOD="circular"
FOLDS="0 1 2 3 4 5 6 7 8 9"
NUM_EXPERIMENTS=3
DATASET_FRACTIONS="1"

# Map array index to classifier × version
# 0=ImageCNN/T25kpc  1=ImageCNN/RAW
# 2=DualCSN/T25kpc   3=DualCSN/RAW
# 4=DualSSN/T25kpc   5=DualSSN/RAW
# 6=SimpleScatterNet/T25kpc 7=SimpleScatterNet/RAW
CLASSIFIER=${ALL_CLASSIFIERS[$((SLURM_ARRAY_TASK_ID / 2))]}
VERSIONS=${ALL_VERSIONS[$((SLURM_ARRAY_TASK_ID % 2))]}

# Output directories for this run
RUN_DIR="/users/mbredber/p2_DCRECLASS/outputs/scratch"
mkdir -p "${RUN_DIR}/figures/"
mkdir -p "${RUN_DIR}/data/logs/"
mkdir -p "${RUN_DIR}/data/models/"
mkdir -p "${RUN_DIR}/data/metrics/"
mkdir -p "/users/mbredber/p2_DCRECLASS/outputs/logs"

# ── Environment ───────────────────────────────────────────────────────────────
source /users/mbredber/p2_DCRECLASS/.venv/bin/activate

export PYTHONPATH=/users/mbredber/.local/lib/python3.11/site-packages:$PYTHONPATH
export PYTHONPATH=/users/mbredber/p2_DCRECLASS/src:$PYTHONPATH

python -c "import dcreclass; print('dcreclass OK')"

echo "========================================"
echo "Job started at: $(date)"
echo "Array task:  ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Classifier:  ${CLASSIFIER}"
echo "Version:     ${VERSIONS}"
echo "Crop mode:   ${CROP_MODE}"
echo "Blur method: ${BLUR_METHOD}"
echo "Run dir:     ${RUN_DIR}"
echo "========================================"

python /users/mbredber/p2_DCRECLASS/scripts/04.train_classifier.py \
    --classifier "${CLASSIFIER}" \
    --versions "${VERSIONS}" \
    --crop-mode "${CROP_MODE}" \
    --blur-method "${BLUR_METHOD}" \
    --run-dir "${RUN_DIR}" \
    --folds ${FOLDS} \
    --num-experiments ${NUM_EXPERIMENTS} \
    --dataset-fractions ${DATASET_FRACTIONS} \
    --noise-level 0.0

echo ""
echo "========================================"
echo "Done at: $(date)"
echo "========================================"
