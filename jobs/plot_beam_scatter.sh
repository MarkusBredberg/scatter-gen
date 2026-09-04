#!/bin/bash -l

#SBATCH --job-name=plot_beam_scatter
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

SCRIPT_PATH="/users/mbredber/p2_DCRECLASS/scripts/plot_beam_scatter.py"
mkdir -p "/users/mbredber/scratch/figures/processing"
mkdir -p "/users/mbredber/p2_DCRECLASS/outputs/logs"

source /users/mbredber/p2_DCRECLASS/.venv/bin/activate
export PYTHONPATH=/users/mbredber/p2_DCRECLASS/src:$PYTHONPATH

python3 -c "import dcreclass; print('dcreclass OK')"

echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================"

python3 "$SCRIPT_PATH"

echo "Outputs: /users/mbredber/scratch/figures/processing/beam_scatter_fig_B1.pdf"
echo "         /users/mbredber/scratch/figures/processing/beam_scatter_fig_B2.pdf"
echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
