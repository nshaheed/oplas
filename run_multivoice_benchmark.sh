#!/usr/bin/bash
#SBATCH --job-name=multivoice
#SBATCH --output=multivoice.%j.out
#SBATCH --error=multivoice.%j.err
#SBATCH --time=4:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg
ml load opencv/4.10.0 libjpeg-turbo


uv run multivoice_benchmark.py -n 32 -k oauvlx7z 00245wks -v v142 v73
