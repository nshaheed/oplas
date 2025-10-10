#!/usr/bin/bash
#SBATCH --job-name=mixing-huber
#SBATCH --output=mixing-huber.%j.out
#SBATCH --error=mixing-huber.%j.err
#SBATCH --time=48:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=128G

ml load system uv gcc/14.2.0 cudnn libsndfile ffmpeg

nvidia-smi

# uv run train.py -d $SCRATCH/musdb18 --num_inner_layers 7 --checkpoint 52kx2dce
# uv run train_sweep.py --resume_run_id jysh86gt
# uv run train_sweep.py --resume_run_id ktp65v1w
uv run train_sweep.py --resume_run_id 3wqlehmx
