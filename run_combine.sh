#!/usr/bin/bash
#SBATCH --job-name=combine
#SBATCH --output=combine.%j.out
#SBATCH --error=combine.%j.err
#SBATCH --time=4:00:00
#SBATCH -p hns,normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=512MB

ml load system uv gcc/14.2.0 cudnn libsndfile ffmpeg opencv/4.10.0 libjpeg-turbo

uv run ffcv_combine_latents_beton.py \
    --in_dir $SCRATCH/mtg-jamendo-ffcv-latents-train \
    --out_path $SCRATCH/mtg-jamendo-ffcv-latents-train \
    --num_stems 64 \
    --num_workers 24
