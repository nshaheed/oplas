#!/usr/bin/bash
#SBATCH --job-name=mixing
#SBATCH --output=mixing.%j.out
#SBATCH --error=mixing.%j.err
#SBATCH --time=6:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=128G

# This can be run like this:
# > sbatch --array=1,3,5 run_array.sh

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg

nvidia-smi

uv run train.py -d $SCRATCH/musdb18 --num_inner_layers ${SLURM_ARRAY_TASK_ID}
