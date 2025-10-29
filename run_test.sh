ml load system
ml load uv
ml load gcc/14.2.0
ml load cudnn
ml load libsndfile
ml load ffmpeg
ml load opencv/4.10.0 libjpeg-turbo

nvidia-smi

uv run train_sweep.py --data_dir $SCRATCH/musdb18 --batch_size 16 --num_stems 4 --num_inner_layers 1 --hidden_dims_scale 1 --learning_rate 1e-4 --val_every 500 --scheduler reduce_on_plateau --max_epochs 30000 --projector_dims 2 --load_frac 0.000025 --val_load_frac 0.001

# uv run train_sweep.py --data_dir $SCRATCH/musdb18 --batch_size 256 --num_stems 64 --num_inner_layers 1 --hidden_dims_scale 1 --learning_rate 1e-5 --val_every 2000 --scheduler 1cycle --max_epochs 2 --max_lr 0.001 --load_frac 0.00048828125 --projector_dims 2

# uv run train_sweep.py --test --data_dir $SCRATCH/musdb18 --batch_size 256 --num_stems 16
