#!/usr/bin/bash
#SBATCH --array=1-10
#SBATCH --job-name=prebake-test
#SBATCH --output=prebake-test.%A_%a.out
#SBATCH --error=prebake-test.%A_%a.err
#SBATCH --time=16:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=20G

ml load system uv gcc/14.2.0 cudnn libsndfile ffmpeg opencv/4.10.0 libjpeg-turbo

nvidia-smi

uv run ffcv_prebaked.py \
    --in_path $SCRATCH/mtg-jamendo-ffcv-test.beton \
    --out_path $SCRATCH/mtg-jamendo-ffcv-latents-test/latents-${SLURM_ARRAY_TASK_ID} \
    --num_workers 4 \
    --num_stems 64 \
    --num_samples 5000
