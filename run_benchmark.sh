#!/usr/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=bechmark.%j.out
#SBATCH --error=benchmark.%j.err
#SBATCH --time=08:00:00
#SBATCH -p hns,normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1GB

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg

uv run benchmark_dataloader.py
