#!/usr/bin/bash
#SBATCH --job-name=sum_loss
#SBATCH --output=sum_loss.%j.out
#SBATCH --error=sum_loss.%j.err
#SBATCH --time=00:30:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=64
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=128G
#SBATCH -C "GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB"

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg
ml load ruse

nvidia-smi

echo "running ruse with 32 train workers and 4 val workers"

ruse -s -t10 --stdout uv run train_sweep.py --data_dir $SCRATCH/musdb18 \
    --adams_beta1 0.8278331554824473 \
    --adams_epsilon 0.00000007938873873216 \
    --cov_coeff 0 \
    --hidden_dims_scale 8 \
    --learning_rate 0.0002050687691449342 \
    --loss pseudo-huber \
    --max_steps 80001 \
    --num_inner_layers 1 \
    --projector_dims 256 \
    --augment \
    --scheduler cosine
