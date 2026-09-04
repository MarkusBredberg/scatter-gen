#!/bin/bash -l

#SBATCH --job-name=plot_results
#SBATCH --account=sk036
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --chdir=/users/mbredber/p2_DCRECLASS
#SBATCH --output=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.out
#SBATCH --error=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

mkdir -p "/users/mbredber/p2_DCRECLASS/outputs/logs"

source /users/mbredber/p2_DCRECLASS/.venv/bin/activate

export PYTHONPATH=/users/mbredber/.local/lib/python3.11/site-packages:$PYTHONPATH
export PYTHONPATH=/users/mbredber/p2_DCRECLASS/src:$PYTHONPATH

echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================"

jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --allow-errors \
    --ExecutePreprocessor.timeout=1200 \
    /users/mbredber/p2_DCRECLASS/notebooks/explore_classification_results.ipynb

echo "========================================"
echo "Done at: $(date)"
echo "========================================"
