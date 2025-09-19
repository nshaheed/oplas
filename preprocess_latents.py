# test to see if we can keep all the latents in memory:

import librosa
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np

import psutil
import os

import math
import random
from tqdm import tqdm
from glob import glob
from music2latent import EncoderDecoder
import time

from oplas.data import MTGJamendoStream

def memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size
    return mem_bytes / (1024 ** 3)  # Convert to MB

# load audio in dataset

class SongDataset(Dataset):
    def __init__(self, data_dir, sample_rate=44100):
        self.data_dir = data_dir
        self.songs_listed = sorted(glob(f"{data_dir}/*/*.mp3"))
        self.sample_rate = sample_rate
        # self.songs_listed = self.songs_listed[:50]

    def __len__(self):
        return len(self.songs_listed)

    def __getitem__(self, idx):
        song = self.songs_listed[idx]
        waveform, sr = librosa.load(song, sr=self.sample_rate)

        # remove second channel etc etc
        # if len(waveform.shape) > 1:
        #     breakpoint()

        # return both waveform and the relative path
        return waveform, os.path.relpath(song, start=self.data_dir)


# Get the index of the current CUDA device
device_idx = torch.cuda.current_device()

# Get total memory (in bytes)
total_memory = torch.cuda.get_device_properties(device_idx).total_memory

# Convert to GB
total_memory_gb = total_memory / (1024**3)

print(f"Total VRAM: {total_memory_gb:.2f} GB")

mem_batch_size = 1
# v100
if total_memory_gb < 4:
    mem_batch_size = 2
elif total_memory_gb < 8:
    mem_batch_size = 4
elif total_memory_gb < 11:
    mem_batch_size = 6
elif total_memory_gb < 16:
    mem_batch_size = 8
elif total_memory_gb < 32:
    mem_batch_size = 12
elif total_memory_gb < 48:
    mem_batch_size = 16
elif total_memroy_gb < 80:
    mem_batch_size = 24




encdec = EncoderDecoder()

data_dir = "/scratch/users/nshaheed/mtg-jamendo/"
save_file = "/scratch/users/nshaheed/mtg-jamendo-latents/latents.npz"
# dataset = SongDataset(data_dir)
dataset = MTGJamendoStream()

dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=4)

progress = tqdm(dataloader)

latents = {}
audios = []

total_time = time.time()

for song in progress:
    song = song.squeeze()
    # path = path[0]

    # waveform, sr = librosa.load(song, sr=sample_rate)

    print(f'{song.shape=}')

    # start = time.time()
    # latent = encdec.encode(song, max_waveform_length=44100*6, max_batch_size=mem_batch_size)
    # print(f'encoding: {time.time() - start:.2f}s')

    # latents[path] = latent.cpu().detach().numpy()

    audios.append(song)
    # print(len(audios))

    # save to file

    # print(f'total time: {time.time() - total_time:.2f}s')
    # total_time = time.time()

    progress.set_postfix(mem=f'{memory_usage():.2f}GB')

np.savez(save_file, **latents)
