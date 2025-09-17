# test to see if we can keep all the latents in memory:

import librosa
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

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

# load audio in dataset

class SongDataset(Dataset):
    def __init__(self, data_dir, sample_rate=44100):
        self.data_dir = data_dir
        self.songs_listed = sorted(glob(f"{data_dir}/*/*.mp3"))
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.songs_listed)

    def __getitem__(self, idx):
        song = self.songs_listed[idx]
        waveform, sr = librosa.load(song, sr=self.sample_rate)

        # remove second channel etc etc
        if len(waveform.shape) > 1:
            breakpoint()

        # return both waveform and the relative path
        return waveform, os.path.relpath(song, start=self.data_dir)


encdec = EncoderDecoder()

data_dir = "/scratch/users/nshaheed/mtg-jamendo/"
dataset = SongDataset(data_dir)

dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=2)

progress = tqdm(dataloader)

latents = []

total_time = time.time()

for song, path in progress:
    song = song.squeeze()
    path = path[0]

    # waveform, sr = librosa.load(song, sr=sample_rate)

    start = time.time()
    latent = encdec.encode(song, max_waveform_length=44100*3, max_batch_size=20).to('cpu')
    print(f'encoding: {time.time() - start:.2f}s')

    latents.append(latent)

    print(f'total time: {time.time() - total_time:.2f}s')
    total_time = time.time()

    progress.set_postfix(mem=f'{memory_usage():.2f}GB')

