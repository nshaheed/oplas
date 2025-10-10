ml load system uv gcc/14.2.0 cudnn libsndfile ffmpeg opencv/4.10.0 libjpeg-turbo

uv run ffcv_combine_latents_beton.py --in_dir /scratch/users/nshaheed/mtg-jamendo-ffcv-latents/ --out_path /scratch/users/nshaheed/mtg-jamendo-ffcv-latents-100.beton --num_workers 1
