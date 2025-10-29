#!/usr/bin/bash
#SBATCH --job-name=plat
#SBATCH --output=plat.%j.out
#SBATCH --error=plat.%j.err
#SBATCH --time=16:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=24G
#SBATCH -C "GPU_MEM:80GB"


ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg
ml load opencv/4.10.0 libjpeg-turbo

nvidia-smi

uv run train_sweep.py --data_dir $SCRATCH/musdb18 \
    --run_from otjkc28v \
    --checkpoint_every 1000 \
    --val_every 1000 \
    --adams_beta1 0.8816049012793831 \
    --adams_epsilon 0.00000000785554863846 \
    --cov_coeff 0.04 \
    --var_coeff 0.71052825253836 \
    --hidden_dims_scale 8 \
    --scheduler reduce_on_plateau \
    --learning_rate 0.0002 \
    --loss pseudo-huber \
    --max_epochs 640 \
    --num_inner_layers 3 \
    --batch_size 256 \
    --num_stems 64 \
    --load_frac 1.0 \
    --projector_dims 256
