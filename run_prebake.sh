#!/usr/bin/bash
#SBATCH --array=1-50
#SBATCH --job-name=prebake
#SBATCH --output=prebake.%A_%a.out
#SBATCH --error=prebake.%A_%a.err
#SBATCH --time=2:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=12G

ml load system uv gcc/14.2.0 cudnn libsndfile ffmpeg opencv/4.10.0 libjpeg-turbo

nvidia-smi

uv run ffcv_prebaked.py \
    --in_path $SCRATCH/mtg-jamendo-ffcv-val.beton \
    --out_path $SCRATCH/mtg-jamendo-ffcv-latents-val-${SLURM_ARRAY_TASK_ID} \
    --num_workers 4 \
    --num_stems 16 \
    --num_samples 5000
