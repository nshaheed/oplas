#!/usr/bin/bash

#SBATCH --job-name=sum_loss
#SBATCH --output=sum_loss.%j.out
#SBATCH --error=sum_loss.%j.err
#SBATCH --time=01:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=32
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=32G
#SBATCH -C "GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB"

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg

uv run train_sweep.py --data_dir /scratch/users/nshaheed/musdb18/ --projector_dims 128 --arch vae --kld_warmup 200 --batch_size 3 --learning_rate 1.2e-3 --load_frac 0.01 --test

# uv run train_sweep.py --data_dir /scratch/users/nshaheed/musdb18/ --projector_dims 16 --kld_warmup 0 --batch_size 3 --learning_rate 1.2e-3 --load_frac 0.01 --test
