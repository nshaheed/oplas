#!/usr/bin/bash
#SBATCH --job-name=v2
#SBATCH --output=v2.%j.out
#SBATCH --error=v2.%j.err
#SBATCH --time=5:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=1024G

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg

nvidia-smi

uv run train_sweep.py --num_chunks 750 \
    --adams_beta1 0.8278331554824473 \
    --adams_epsilon 0.00000007938873873216 \
    --cov_coeff 0 \
    --hidden_dims_scale 8 \
    --learning_rate 0.0002050687691449342 \
    --loss pseudo-huber \
    --max_steps 120001 \
    --num_inner_layers 1 \
    --projector_dims 256 \
    --augment \
    --scheduler cosine \
    --val_every 500 \
    --num_stems 16 \
    --batch_size 64 \
    --checkpoint_every 2000

# uv run train_sweep.py --test --num_chunks 10 \
#     --adams_beta1 0.8278331554824473 \
#     --adams_epsilon 0.00000007938873873216 \
#     --cov_coeff 0 \
#     --hidden_dims_scale 8 \
#     --learning_rate 0.0002050687691449342 \
#     --loss pseudo-huber \
#     --max_steps 80001 \
#     --num_inner_layers 1 \
#     --projector_dims 256 \
#     --augment \
#     --scheduler cosine
