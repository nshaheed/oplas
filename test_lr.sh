#!/usr/bin/bash
#SBATCH --job-name=lrtest
#SBATCH --output=lrtest.%j.out
#SBATCH --error=lrtest.%j.err
#SBATCH --time=0:10:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=32G
#SBATCH -C "GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB"

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg
ml load opencv/4.10.0 libjpeg-turbo

nvidia-smi


uv run lr_range.py --data_dir $SCRATCH/musdb18 \
    --adams_beta1 0.8816049012793831 \
    --adams_epsilon 0.00000000785554863846 \
    --cov_coeff 0.04 \
    --var_coeff 0.71052825253836 \
    --hidden_dims_scale 8 \
    --num_inner_layers 3 \
    --batch_size 512 \
    --num_stems 16 \
    --projector_dims 256
