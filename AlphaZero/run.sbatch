#!/bin/bash
#SBATCH -J ChickenZero
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -q inferno
#SBATCH -o Output-%j.out
#SBATCH -e Error-%j.err

# 1. Load Environment
module load pytorch  # Load standard PACE pytorch module

# 2. Debug Info
echo "Running on node: $(hostname)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 3. Run the Training Script
# We assume the zip file was extracted such that 'engine' and '3600-agents' are visible.
# We set PYTHONPATH to the current directory so python can find 'engine'.

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the script
# We run as a module to handle relative imports if needed, or direct file.
python3 3600-agents/AlphaZero/train.py