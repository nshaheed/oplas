#!/usr/bin/bash
#SBATCH --array=1-16
#SBATCH --job-name=prebake-val
#SBATCH --output=prebake-val.%A_%a.out
#SBATCH --error=prebake-val.%A_%a.err
#SBATCH --time=16:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=20G

ml load system uv gcc/14.2.0 cudnn libsndfile ffmpeg opencv/4.10.0 libjpeg-turbo

nvidia-smi

STEMS=2

mkdir -p $SCRATCH/mtg-jamendo-ffcv-latents-val-${STEMS}

uv run ffcv_prebaked.py \
    --in_path $SCRATCH/mtg-jamendo-ffcv-val.beton \
    --out_path $SCRATCH/mtg-jamendo-ffcv-latents-val-${STEMS}/latents-${SLURM_ARRAY_TASK_ID} \
    --num_workers 4 \
    --num_stems ${STEMS} \
    --num_samples 15625
