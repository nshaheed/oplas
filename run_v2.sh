#!/usr/bin/bash
#SBATCH --job-name=v2
#SBATCH --output=v2.%j.out
#SBATCH --error=v2.%j.err
#SBATCH --time=24:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=128G
#SBATCH -C "GPU_MEM:48GB|GPU_MEM:80GB"

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg
ml load opencv/4.10.0 libjpeg-turbo

uv run train_sweep.py --data_dir $SCRATCH/musdb18 \
    --sweep_id nshaheed-stanford-university/oplas/o2pkcihg
