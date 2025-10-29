#!/usr/bin/bash
#SBATCH --array=0
#SBATCH --job-name=multivoice
#SBATCH --output=multivoice.%A_%a.out
#SBATCH --error=multivoice.%A_%a.err
#SBATCH --time=16:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G

ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg
ml load opencv/4.10.0 libjpeg-turbo


#      1/256      1/128      1/64       1/32       1/16       1/8        1/4        1/2
keys=("s52zbb9h" "0t0ifbvt" "d8gkgiu9" "rrx9gt7v" "otjkc28v" "ek6cahrw" "t2ib5c80" "qahnouow")

key=${keys[$SLURM_ARRAY_TASK_ID]}

echo "Running task ${SLURM_ARRAY_TASK_ID} with key=${key}"

# uv run multivoice_benchmark.py -k ${key} -v latest -n ${SLURM_ARRAY_TASK_ID}

for ((n=1; n<8; n++)); do
    echo "Starting run with n=$n"
    uv run multivoice_benchmark.py -k "$key" -v latest -n "$n"
done

for ((n=8; n<16; n+=2)); do
    echo "Starting run with n=$n"
    uv run multivoice_benchmark.py -k "$key" -v latest -n "$n"
done

for ((n=16; n<32; n+=4)); do
    echo "Starting run with n=$n"
    uv run multivoice_benchmark.py -k "$key" -v latest -n "$n"
done

for ((n=32; n<64; n+=8)); do
    echo "Starting run with n=$n"
    uv run multivoice_benchmark.py -k "$key" -v latest -n "$n"
done

for ((n=64; n<128; n+=16)); do
    echo "Starting run with n=$n"
    uv run multivoice_benchmark.py -k "$key" -v latest -n "$n"
done
