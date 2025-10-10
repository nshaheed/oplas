#!/usr/bin/bash
#SBATCH --job-name=ffcv
#SBATCH --output=ffcv.%j.out
#SBATCH --error=ffcv.%j.err
#SBATCH --time=15:00:00
#SBATCH -p hns,normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1GB


ml load system uv gcc/14.2.0 cudnn libsndfile ffmpeg opencv/4.10.0 libjpeg-turbo


uv run make_ffcv.py --audio_dir $SCRATCH/mtg-jamendo-wav --out_path $SCRATCH/mtg-jamendo-ffcv --num_workers 0
