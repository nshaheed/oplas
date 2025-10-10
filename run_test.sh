ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg
ml load opencv/4.10.0 libjpeg-turbo

nvidia-smi

uv run train_sweep.py --data_dir $SCRATCH/musdb18 --batch_size 32 --num_stems 16 --num_inner_layers 1 --hidden_dims_scale 8 --learning_rate 1e-5 --val_every 2000 --scheduler 1cycle --max_epochs 2

# uv run train_sweep.py --test --data_dir $SCRATCH/musdb18 --batch_size 256 --num_stems 16
