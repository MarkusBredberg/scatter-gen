#!/bin/bash -l

#SBATCH --job-name=power_spectrum_comparison
#SBATCH --account=sk036
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:15:00
#SBATCH --chdir=/users/mbredber/p2_DCRECLASS
#SBATCH --output=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.out
#SBATCH --error=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

mkdir -p "/users/mbredber/scratch/figures/processing"
mkdir -p "/users/mbredber/p2_DCRECLASS/outputs/logs"

source /users/mbredber/p2_DCRECLASS/.venv/bin/activate
export PYTHONPATH=/users/mbredber/p2_DCRECLASS/src:$PYTHONPATH

echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================"

python3 tests/power_spectrum_comparison.py

echo "Output: /users/mbredber/scratch/figures/processing/power_spectrum_comparison.png"
echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
