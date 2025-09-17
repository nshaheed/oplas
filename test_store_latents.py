# test to see if we can keep all the latents in memory:

import librosa
import torch
import torchaudio

import psutil
import os

import random
from tqdm import tqdm
from glob import glob
from music2latent import EncoderDecoder
import time

def memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size
    return mem_bytes / (1024 ** 3)  # Convert to MB


data_dir="/scratch/users/nshaheed/mtg-jamendo"
data_dir="/scratch/users/nshaheed/mtg-jamendo/"
latents = []

songs_listed = sorted(glob(f"{data_dir}/*/*.mp3"))
# songs_listed = sorted(glob(f"{data_dir}/*/*"))
sample_rate = 44100

# print(songs_listed)

encdec = EncoderDecoder(device='cpu')

progress = tqdm(songs_listed)

for song in progress:
    waveform, sr = librosa.load(song, sr=sample_rate)

    if len(waveform.shape) > 1:
        breakpoint()

    start = time.time()
    latent = encdec.encode(waveform, max_waveform_length=44100*1).to('cpu')
    print(f'encoding: {time.time() - start:.2f}s')

    latents.append(latent)

    progress.set_postfix(mem=f'{memory_usage():.2f}GB')
