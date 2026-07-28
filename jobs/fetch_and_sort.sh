#!/bin/bash -l

#SBATCH --job-name=fetch_sort
#SBATCH --account=sk036
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --chdir=/users/mbredber/p2_DCRECLASS
#SBATCH --output=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.out
#SBATCH --error=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

mkdir -p "/users/mbredber/p2_DCRECLASS/outputs/logs"

source /users/mbredber/p2_DCRECLASS/.venv/bin/activate

export PYTHONPATH=/users/mbredber/p2_DCRECLASS/src:$PYTHONPATH

python3 -c "import requests, bs4; print('fetch deps OK')"

echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================"

echo "--- Step 1: Download missing PSZ2 sources ---"
python3 -u /users/mbredber/p2_DCRECLASS/scripts/01.rsync_PSZ2.py

echo "--- Step 2: Categorise missing sources ---"
python3 -u /users/mbredber/p2_DCRECLASS/scripts/02.categorise_PSZ2.py --symlink

echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
