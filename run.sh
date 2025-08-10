#!/usr/bin/bash
#SBATCH --job-name=mixing
#SBATCH --output=mixing.%j.out
#SBATCH --error=mixing.%j.err
#SBATCH --time=12:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=128G

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg
# ml load python/3.12.1
# ml load py-pytorch/2.4.1_py312
# ml load py-scipystack
# ml load py-wandb

nvidia-smi

uv run train.py -d $SCRATCH/musdb18
