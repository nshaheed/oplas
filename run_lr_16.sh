#!/usr/bin/bash
#SBATCH --job-name=cyclic16
#SBATCH --output=cyclic16.%j.out
#SBATCH --error=cyclic16.%j.err
#SBATCH --time=8:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg
ml load opencv/4.10.0 libjpeg-turbo

uv run train_sweep.py --data_dir $SCRATCH/musdb18 \
    --checkpoint_every 3500 \
    --val_every 3500 \
    --adams_beta1 0.8683258839505463 \
    --adams_epsilon 0.0000000125896464951 \
    --cov_coeff 0.04 \
    --var_coeff 0.9672151406290842 \
    --hidden_dims_scale 16 \
    --scheduler cyclic \
    --base_lr 4.5e-6 --max_lr 3e-4 \
    --loss pseudo-huber \
    --max_steps 500 \
    --num_inner_layers 1 \
    --batch_size 64 \
    --num_stems 16 \
    --projector_dims 256
