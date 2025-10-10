#!/usr/bin/bash
#SBATCH --job-name=sum_loss
#SBATCH --output=sum_loss.%j.out
#SBATCH --error=sum_loss.%j.err
#SBATCH --time=14:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=32
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=32G
#SBATCH -C "GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB"

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg

nvidia-smi

uv run train_sweep.py --sweep_id nshaheed-stanford-university/oplas/o2pkcihg
