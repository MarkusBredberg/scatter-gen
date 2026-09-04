#!/bin/bash -l

#SBATCH --job-name=crop_comparison
#SBATCH --account=sk036
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --chdir=/users/mbredber/p2_DCRECLASS
#SBATCH --output=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.out
#SBATCH --error=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4

SCRIPT_PATH="/users/mbredber/p2_DCRECLASS/scripts/03.create_processed_images.py"
mkdir -p "/users/mbredber/scratch/figures/processing"
mkdir -p "/users/mbredber/p2_DCRECLASS/outputs/logs"

source /users/mbredber/p2_DCRECLASS/.venv/bin/activate
export PYTHONPATH=/users/mbredber/p2_DCRECLASS/src:$PYTHONPATH

python3 -c "import dcreclass; print('dcreclass OK')"

echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================"

# To use a specific source, add: --crop-comp-source PSZ2G...
# To use a random DE source, omit --crop-comp-source (seed controlled by --comp-seed)
python3 "$SCRIPT_PATH" \
    --crop-comparison \
    --no-annotate \
    --no-montage \
    --scales 25,50,100 \
    --blur-method circular \
    --crop-mode beam_crop \
    --crop-comp-fov 800 \
    --crop-comp-figsize 14,6 \
    --comp-seed 10

echo "Output: /users/mbredber/scratch/figures/processing/crop_strategy_comparison.png"
echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
