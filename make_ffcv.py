import os
import math
import torch
import torchaudio
import numpy as np
import soundfile as sf
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
from glob import glob
from torch.utils.data import Dataset
from tqdm import tqdm

CHUNK_SIZE = 2**18  # Number of samples per chunk (262144)
SAMPLE_RATE = 44100


def load_audio(path, sample_rate=SAMPLE_RATE):
    """
    Load an audio file as a mono tensor at a given sample rate.
    """
    wv, orig_sr = sf.read(path, dtype="float32", always_2d=True)
    wv = torch.from_numpy(wv)

    # Make mono
    wv = wv[:, 0]

    # Resample if needed
    if orig_sr != sample_rate:
        wv = torchaudio.functional.resample(wv, orig_sr, sample_rate)

    return wv


class AudioChunkDataset(Dataset):
    def __init__(self, audio_dir, chunk_size=CHUNK_SIZE, sample_rate=SAMPLE_RATE):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.files = sorted(glob(f"{audio_dir}/*/*.wav") + glob(f"{audio_dir}/*/*.mp3"))
        # self.files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith((".wav", ".flac", ".mp3"))]

        # Precompute chunks metadata: (filename, chunk_index)
        self.index_map = []
        for file in (pbar := tqdm(self.files)):
            pbar.set_description("loading metadata")
            info = sf.info(file)
            orig_sr = info.samplerate
            num_samples = (
                int(info.frames * (sample_rate / orig_sr))
                if orig_sr != sample_rate
                else info.frames
            )
            num_chunks = math.ceil(num_samples / chunk_size)
            for i in range(num_chunks):
                self.index_map.append((file, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file, chunk_idx = self.index_map[idx]
        wv = load_audio(file, self.sample_rate)

        start = chunk_idx * self.chunk_size
        end = start + self.chunk_size

        chunk = torch.zeros(self.chunk_size, dtype=torch.float32)
        available = wv[start:end]
        chunk[: len(available)] = available

        print(f'{chunk.contiguous().numpy().shape=}')
        return (chunk.contiguous().numpy(),)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir", type=str, required=True, help="Directory with audio files"
    )
    parser.add_argument(
        "--out_path", type=str, required=True, help="Path to save ffcv dataset (.beton)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for parallel writing",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Portion of dataset to be split into the val chunk",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.01,
        help="Portion of dataset to be split into the test chunk",
    )
    args = parser.parse_args()

    dataset = AudioChunkDataset(args.audio_dir)

    train_split = 1 - args.val_split - args.test_split
    val_split, test_split = args.val_split, args.test_split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_split, val_split, test_split]
    )

    writer_test = DatasetWriter(
        f"{args.out_path}-test.beton",
        {"audio": NDArrayField(shape=(CHUNK_SIZE,), dtype=np.dtype("float32"))},
        num_workers=args.num_workers,
    )

    writer_test.from_indexed_dataset(test_dataset)

    writer_val = DatasetWriter(
        f"{args.out_path}-val.beton",
        {"audio": NDArrayField(shape=(CHUNK_SIZE,), dtype=np.dtype("float32"))},
        num_workers=args.num_workers,
    )

    writer_val.from_indexed_dataset(val_dataset)

    writer_train = DatasetWriter(
        f"{args.out_path}-train.beton",
        {"audio": NDArrayField(shape=(CHUNK_SIZE,), dtype=np.dtype("float32"))},
        num_workers=args.num_workers,
    )

    writer_train.from_indexed_dataset(train_dataset)
