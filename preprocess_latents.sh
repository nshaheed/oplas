#!/usr/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess.%j.out
#SBATCH --error=preprocess.%j.err
#SBATCH --time=18:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=128G
#SBATCH -C "GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB"

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg

nvidia-smi

uv run preprocess_latents.py
