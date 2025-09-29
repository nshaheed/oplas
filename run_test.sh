ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg
ml load opencv/4.10.0 libjpeg-turbo

nvidia-smi

uv run train_sweep.py --data_dir $SCRATCH/musdb18 --batch_size 256 --num_stems 16
