#!/usr/bin/bash
#SBATCH --job-name=combine
#SBATCH --output=combine.%j.out
#SBATCH --error=combine.%j.err
#SBATCH --time=4:00:00
#SBATCH -p hns,normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=512MB

ml load system uv gcc/14.2.0 cudnn libsndfile ffmpeg opencv/4.10.0 libjpeg-turbo

STEMS=32

uv run ffcv_combine_latents_beton.py \
    --in_dir $SCRATCH/mtg-jamendo-ffcv-latents-val-${STEMS} \
    --out_path $SCRATCH/mtg-jamendo-ffcv-latents-val-${STEMS} \
    --num_stems ${STEMS} \
    --num_workers 4
