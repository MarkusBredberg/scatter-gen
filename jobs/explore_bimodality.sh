#!/bin/bash -l

#SBATCH --job-name=explore_bimodality
#SBATCH --account=sk036
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --chdir=/users/mbredber/p2_DCRECLASS
#SBATCH --output=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.out
#SBATCH --error=/users/mbredber/p2_DCRECLASS/outputs/logs/sbatchrun-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

mkdir -p "/users/mbredber/scratch/figures/processing"
mkdir -p "/users/mbredber/p2_DCRECLASS/outputs/logs"

source /users/mbredber/p2_DCRECLASS/.venv/bin/activate
export PYTHONPATH=/users/mbredber/p2_DCRECLASS/src:$PYTHONPATH

python3 -c "import dcreclass; print('dcreclass OK')"

echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================"

echo ""
echo "--- Step 1: beam scatter plots — theoretical blur ---"
python3 scripts/plot_beam_scatter.py

echo ""
echo "--- Step 2: beam scatter plots — actual blur (from processed FITS headers) ---"
python3 scripts/plot_beam_scatter.py --actual-blur

echo ""
echo "--- Step 3: ratio sampler — theoretical blur ---"
python3 scripts/beam_ratio_sampler.py --scale 50 --n 6

echo ""
echo "--- Step 4: ratio sampler — actual blur ---"
python3 scripts/beam_ratio_sampler.py --scale 50 --n 6 --actual-blur

echo ""
echo "Outputs:"
echo "  beam_scatter_fig_B1_theoretical.pdf / beam_scatter_fig_B1_actual.pdf"
echo "  beam_scatter_fig_B2_theoretical.pdf / beam_scatter_fig_B2_actual.pdf"
echo "  beam_ratio_histogram_50kpc_theoretical.pdf / beam_ratio_histogram_50kpc_actual.pdf"
echo "  beam_ratio_sampler_50kpc_theoretical.pdf   / beam_ratio_sampler_50kpc_actual.pdf"
echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
