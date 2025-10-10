#!/usr/bin/bash
#SBATCH --job-name=cache
#SBATCH --output=cache.%j.out
#SBATCH --error=cache.%j.err
#SBATCH --time=08:00:00
#SBATCH -p hns,normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8GB

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg

uv run cache_audio.py
