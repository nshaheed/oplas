#!/usr/bin/bash
#SBATCH --array=512,1024
#SBATCH --job-name=dataset
#SBATCH --output=dataset.%A_%a.out
#SBATCH --error=dataset.%A_%a.err
#SBATCH --time=30:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=32G
#SBATCH -C "GPU_MEM:80GB"

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg
ml load opencv/4.10.0 libjpeg-turbo

nvidia-smi

LOAD_FRAC=$(echo "scale=10; 1/${SLURM_ARRAY_TASK_ID}" | bc)
EPOCHS=$(echo $((SLURM_ARRAY_TASK_ID * 40)))

echo "Running with load frac of ${LOAD_FRAC} for ${EPOCHS} epochs"

uv run train_sweep.py --data_dir $SCRATCH/musdb18 \
    --checkpoint_every 1000 \
    --val_every 1000 \
    --adams_beta1 0.8816049012793831 \
    --adams_epsilon 0.00000000785554863846 \
    --cov_coeff 0.04 \
    --var_coeff 0.71052825253836 \
    --hidden_dims_scale 8 \
    --scheduler 1cycle \
    --base_lr 3e-5 --max_lr 0.004 \
    --loss pseudo-huber \
    --max_epochs ${EPOCHS} \
    --num_inner_layers 3 \
    --batch_size 256 \
    --num_stems 64 \
    --load_frac ${LOAD_FRAC} \
    --projector_dims 256
