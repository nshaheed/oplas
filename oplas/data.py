import ctypes
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob
import sys

import soundfile as sf
import numpy as np
import stempeg
import torch
import torchaudio
from torch.utils.data import Dataset, IterableDataset, ChainDataset
from tqdm import tqdm, trange

from torch_audiomentations import (
    Compose,
    AddColoredNoise,
    AddBackgroundNoise,
    ApplyImpulseResponse,
    PitchShift,
    PolarityInversion,
    Gain,
    BandPassFilter,
    BandStopFilter,
    HighPassFilter,
    LowPassFilter,
)

# from tqdm.contrib.concurrent import process_map # doesn't work great w/ jupyter
from tqdm.contrib.concurrent import process_map


def augment_effects(
    waveforms: torch.Tensor,
    sample_rate: int = 44100,
    use_noise: bool = True,
) -> torch.Tensor:
    """
    Apply randomized augmentations to audio waveforms using torch-audiomentations.


    Args:
    waveforms (torch.Tensor): Input tensor of shape [batch_size, stems, time, channels]
    sample_rate (int): Audio sample rate used for augmentations
    use_noise (bool): Enable various noise augmentations


    Returns:
    torch.Tensor: Augmented tensor of the same shape
    """
    stems, time, channels = waveforms.shape

    # Rearrange into [batch*stems*channels, 1, time] for torch-audiomentations
    waveforms_reshaped = (
        waveforms.permute(0, 2, 1).reshape(  # [batch, stems, channels, time]
            -1, 1, time
        )  # [batch*stems*channels, 1, time]
    )

    # waveforms_reshaped = (
    #     waveforms.permute(0, 1, 3, 2).reshape(  # [batch, stems, channels, time]
    #         -1, 1, time
    #     )  # [batch*stems*channels, 1, time]
    # )

    # Build augmentation pipeline
    transforms = []

    mode = "per_channel"
    ot = "tensor"  # output type

    if use_noise:
        transforms.append(
            AddColoredNoise(
                min_snr_in_db=10, max_snr_in_db=30, p=0.5, mode=mode, output_type=ot
            )
        )
        # transforms.append(AddBackgroundNoise(background_paths="./backgrounds", p=0.5))
        # transforms.append(ApplyImpulseResponse(ir_paths="./impulse_responses", p=0.5))

    transforms.extend(
        [
            PitchShift(
                min_transpose_semitones=-2,
                max_transpose_semitones=2,
                sample_rate=sample_rate,
                p=0.5,
                mode=mode,
                output_type=ot,
            ),
            PolarityInversion(p=0.3, mode=mode, output_type=ot),
            BandPassFilter(p=0.3, mode=mode, output_type=ot),
            BandStopFilter(p=0.3, mode=mode, output_type=ot),
            HighPassFilter(p=0.3, mode=mode, output_type=ot),
            LowPassFilter(p=0.3, mode=mode, output_type=ot),
        ]
    )

    augment = Compose(transforms, output_type=ot)

    # Apply augmentations
    augmented = augment(waveforms_reshaped, sample_rate=sample_rate)

    # Reshape back to [batch_size, stems, time, channels]
    augmented = augmented.reshape(stems, channels, time).permute(0, 2, 1)

    return augmented


class RandomMixDataset(IterableDataset):
    def __init__(self, datasets, probs=None):
        super().__init__()
        self.datasets = datasets
        self.probs = probs or [1.0 / len(datasets)] * len(datasets)

    def __iter__(self):
        # Create iterators for each dataset
        iters = [iter(ds) for ds in self.datasets]

        while iters:
            # Pick one dataset according to probs
            choice = random.choices(range(len(iters)), weights=self.probs, k=1)[0]
            try:
                yield next(iters[choice])
            except StopIteration:
                # Remove exhausted iterator
                del iters[choice]
                del self.probs[choice]
                if self.probs:
                    # Renormalize probs
                    total = sum(self.probs)
                    self.probs = [p / total for p in self.probs]


class MTGJamendo(Dataset):
    """Loads pre-encoded audio files from a .npz file.

    While the previous approach involved loading equal-sized chunks
    from several audio files, this dataset will preload everything
    in-memory and then only return the chunk from one audio file.

    """

    def __init__(
        self,
        data_file="/scratch/users/nshaheed/mtg-jamendo-latents/latents.npz",
        chunk_size=65,
        load_frac=1.0,
        # augment=False,
        debug=False,
    ):
        self.load_frac = load_frac  # not doing anything with this atm
        self.chunk_size = chunk_size
        self.data = np.load(data_file)
        self.keys = list(self.data.keys())
        self.debug = debug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.debug is True:
            print(f"{len(self.songs_listed)=}")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            seed = random.randint(0, 2**32 - 1)
        else:
            # Unique seed per worker
            seed = worker_info.seed

        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        rng = random.Random(seed)

        key = self.keys[idx]
        value = self.data[key]

        value = torch.from_numpy(value)

        start = rng.randint(0, value.shape[-1] - self.chunk_size)
        chunk = value[:, :, start : start + self.chunk_size]

        return chunk[0]


class MTGJamendoStreamSingle(Dataset):
    """Loads chuck from single file from audio dataset

    this dataset doesn't have a concept of stems and will let that be
    handled at the dataloader level

    """

    def __init__(
        self,
        data_dir="/scratch/users/nshaheed/mtg-jamendo",
        # data_dir="/scratch/nshaheed/mtg-jamendo",
        chunk_size=2**18,
        sample_rate=44100,
        load_frac=1.0,
        augment=False,
        debug=False,
    ):
        # self.songs_listed = sorted(glob(f"{data_dir}/*/*.mp3"))
        self.songs_listed = sorted(glob(f"{data_dir}/*/*.wav"))
        self.chunk_size = chunk_size
        self.debug = debug
        self.augment = augment
        self.sample_rate = sample_rate

        if load_frac < 1.0:
            keep_n = int(len(self.songs_listed) * load_frac)
            keep_n = max(keep_n, 1)
            self.songs_listed = random.sample(self.songs_listed, keep_n)

        self.song_metadata = []
        get_metadata = tqdm(self.songs_listed)
        get_metadata.set_description("loading audio metadata")
        for path in get_metadata:
            info = sf.info(path)
            self.song_metadata.append(
                {
                    "path": path,
                    "samplerate": info.samplerate,
                    "duration": info.duration,
                    "num_samples": int(info.samplerate * info.duration),
                }
            )

    def __len__(self):
        return len(self.songs_listed)

    def __getitem__(self, idx):
        if self.debug is True:
            print(f"{len(self.songs_listed)=}")

        # random bs involving seeding different rngs
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.seed if worker_info else random.randint(0, 2**32 - 1)
        rng = random.Random(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)

        song_path = self.songs_listed[idx]
        info = self.song_metadata[idx]
        samplerate = info["samplerate"]
        duration = info["duration"]
        length = int(samplerate * duration)
        chunk_size_dur = self.chunk_size / self.sample_rate
        chunk_size_file = int(chunk_size_dur * samplerate) + samplerate

        rand_start = torch.randint(length - chunk_size_file, size=(1,)).item()
        wv, _ = sf.read(
            song_path,
            frames=chunk_size_file,
            start=rand_start,
            stop=None,
            dtype="float32",
            always_2d=True,
        )
        wv = torch.from_numpy(wv)
        if wv.shape[-1] == 1:
            wv = torch.cat([wv, wv], dim=1)
        wv = wv[:, :2]
        wv = wv.permute(1, 0)

        # if not stereo:
        wv = wv[torch.randint(wv.shape[0], size=(1,)).item(), :]

        # rms = torch.sqrt(torch.mean(wv**2))
        # if rms < self.rms_min:
        #     idx = torch.randint(self.tot_samples, size=(1,)).item()
        #     return self.__getitem__(idx)
        ## -----------
        # return wv

        # TODO actually handle things properly
        # breakpoint()
        wv = wv[: self.chunk_size]
        return wv


class MTGJamendoCache(Dataset):
    """Loads pre-tensored, full audio files (generated with cache_audio.py)

    This stores everything in-memory and should be very-fast

    """

    def __init__(
        self,
        data_dir="/scratch/users/nshaheed/mtg-jamendo-cache",
        # data_dir="/scratch/nshaheed/mtg-jamendo",
        chunk_size=2**18,
        sample_rate=44100,
        num_chunks=None,  # if None, load all
        augment=False,
        debug=False,
    ):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.debug = debug
        self.augment = augment

        shard_paths = sorted(glob(os.path.join(data_dir, "cache_*.pt")))
        if shard_paths:
            print(f"Found {len(shard_paths)} shards in {data_dir}.")

            # truncate
            if num_chunks is not None:
                shard_paths = shard_paths[:num_chunks]

            print(f"Loading {len(shard_paths)} shard(s) from cache...")
            tensors = []
            paths = tqdm(shard_paths)
            paths.set_description("loading shards")
            for path in paths:
                tensors.extend(torch.load(path))
            self.songs = tensors
        else:
            print(f"No shards found in {data_dir}, Dataset will be empty")
            self.songs = []

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        if self.debug is True:
            print(f"{len(self.songs_listed)=}")

        # random bs involving seeding different rngs
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.seed if worker_info else random.randint(0, 2**32 - 1)
        rng = random.Random(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)

        # TODO handle short audio
        song = self.songs[idx]

        if song.shape[0] < self.chunk_size:
            chunk = torch.nn.functional.pad(song, (0, self.chunk_size - song.shape[0]))

        else:
            start = rng.randint(0, song.shape[0] - self.chunk_size)
            chunk = song[start : start + self.chunk_size]

        # return chunk
        return chunk


class MTGJamendoStream(IterableDataset):
    def __init__(
        self,
        data_dir="/scratch/users/nshaheed/mtg-jamendo",
        # data_dir="/scratch/nshaheed/mtg-jamendo",
        chunk_size=2**18,
        sample_rate=44100,
        max_num_stems=5,
        load_frac=1.0,
        augment=False,
        debug=False,
    ):
        self.songs_listed = sorted(glob(f"{data_dir}/*/*.mp3"))

        if load_frac < 1.0:
            keep_n = int(len(self.songs_listed) * load_frac)
            keep_n = max(keep_n, 1)
            self.songs_listed = random.sample(self.songs_listed, keep_n)

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.debug = debug
        self.max_num_stems = max_num_stems
        self.augment = augment

        # Precompute song lengths in samples
        # self.song_lengths = []
        # for song_path in self.songs_listed:
        #     info = stempeg.Info(song_path)
        #     duration_sec = info.duration(0)
        #     num_samples = int(duration_sec * info.sample_rate(0))
        #     self.song_lengths.append(num_samples)

    def __iter__(self):
        if self.debug is True:
            print(f"{len(self.songs_listed)=}")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            seed = random.randint(0, 2**32 - 1)
        else:
            # Unique seed per worker
            seed = worker_info.seed

        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        rng = random.Random(seed)

        while True:  # infitie stream of da music
            stems = []
            chunk_size_dur = self.chunk_size / self.sample_rate

            # get random # of stems
            # max_stems = random.randrange(1,self.max_num_stems)+1
            max_stems = self.max_num_stems

            while len(stems) < max_stems:
                song_idx = rng.randint(0, len(self.songs_listed) - 1)
                song_path = self.songs_listed[song_idx]

                ## try with sf
                info = sf.info(song_path)
                samplerate = info.samplerate
                duration = info.duration
                length = int(samplerate * duration)
                chunk_size_file = int(chunk_size_dur * samplerate) + samplerate

                rand_start = torch.randint(length - chunk_size_file, size=(1,)).item()
                wv, _ = sf.read(
                    song_path,
                    frames=chunk_size_file,
                    start=rand_start,
                    stop=None,
                    dtype="float32",
                    always_2d=True,
                )
                wv = torch.from_numpy(wv)
                if wv.shape[-1] == 1:
                    wv = torch.cat([wv, wv], dim=1)
                wv = wv[:, :2]
                wv = wv.permute(1, 0)

                # if not stereo:
                wv = wv[torch.randint(wv.shape[0], size=(1,)).item(), :]

                # rms = torch.sqrt(torch.mean(wv**2))
                # if rms < self.rms_min:
                #     idx = torch.randint(self.tot_samples, size=(1,)).item()
                #     return self.__getitem__(idx)
                ## -----------
                # return wv

                # TODO actually handle things properly
                # breakpoint()
                wv = wv[: self.chunk_size]
                stems.append(wv)
                continue

                waveform = waveform[
                    : self.chunk_size, :
                ]  # a hack to handle oddly shaped data sizes

                if waveform.shape[1] == 1:
                    waveform = waveform.repeat(1, 2)
                    # breakpoint()

                # scale audio
                waveform = (1.0 / max_stems) * waveform
                stems.append(waveform)

            # for stem in stems:
            #     if stem.shape != torch.Size([262144,2]):
            #         print("NO")
            stems_tensor = torch.stack(stems)

            if self.augment:
                stems_tensor = augment_effects(stems_tensor, self.sample_rate)

            # if stems_tensor.shape != [max_stems, 262144,2]:
            #     breakpoint()
            yield stems_tensor


class StemChunk(IterableDataset):
    """
    Infinite stream of random audio chunks from MUSDB18 stems using preloading.

    This reads MUSDB18 stems files in .mp4 format. The contents of these are given in MUSDB18 docs:
    0 - The mixture,    <--- note we're not going to use this
    1 - The drums,
    2 - The bass,
    3 - The rest of the accompaniment,
    4 - The vocals.
    """

    def __init__(
        self,
        subset="train",
        data_dir="/home/shawley/datasets/musdb18-stems",
        chunk_size=2**18,  # number of samples per chunk
        sample_rate=44100,
        load_frac=1.0,  # fraction of dataset to use
        debug=False,
    ):
        search_dir = os.path.join(data_dir, subset)
        self.songs_listed = sorted(glob(f"{search_dir}/*.mp4"))
        if load_frac < 1.0:
            keep_n = max(1, int(len(self.songs_listed) * load_frac))
            keep_n = max(keep_n, 1)
            self.songs_listed = random.sample(self.songs_listed, keep_n)

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.debug = debug
        self.subset = subset

        # Precompute song lengths in samples
        self.song_lengths = []
        for song_path in self.songs_listed:
            info = stempeg.Info(song_path)
            duration_sec = info.duration(0)
            num_samples = int(duration_sec * info.sample_rate(0))
            self.song_lengths.append(num_samples)

        if debug:
            print(f"{subset}: {len(self.songs_listed)} songs listed.")
            print(
                f"Chunk size: {chunk_size} samples ({chunk_size / sample_rate:.2f} sec)"
            )
            print(f"{len(self.songs_listed)=}")

        self.preload(load_frac, debug)

    def preload(self, load_frac=1.0, debug=False):
        print(f"{self.subset}: Preloading songs...")
        self.songs = []

        for i, song_name in tqdm(
            enumerate(self.songs_listed),
            desc="loading audio",
            file=sys.stdout,
            disable=not sys.stdout.isatty(),
        ):
            self.songs.append(self.load_song(i))

    def load_song(self, idx, start=0, duration=None, debug=False):
        "loads one song file"
        if type(idx) is int:
            song_file = self.songs_listed[idx]
        elif type(idx) is str:
            song_file = idx
        else:
            print("Unsupported datatype = ", type(idx))

        if debug or self.debug:
            print(f"{self.subset}: Loading {song_file}", flush=True)

        data, sample_rate = stempeg.read_stems(
            song_file, sample_rate=self.sample_rate, start=start, duration=duration
        )
        data = torch.tensor(data, dtype=torch.float32)
        if debug:
            print(
                f"load_song {idx}: {self.songs_listed[idx]}: data.shape = ", data.shape
            )
        song_dict = {
            "name": song_file,
            "data": data,
            "sample_rate": sample_rate,
            "length": data.shape[1],
        }
        return song_dict

    def __iter__(self):
        if self.debug is True:
            print(f"{len(self.songs_listed)=}")

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            seed = random.randint(0, 2**32 - 1)
        else:
            # Unique seed per worker
            seed = worker_info.seed

        rng = random.Random(seed)

        while True:
            # Pick a random song
            song_idx = rng.randint(0, len(self.songs_listed) - 1)
            song = self.songs[song_idx]
            # song_path = self.songs_listed[song_idx]
            # song_len = self.song_lengths[song_idx]

            data = song["data"]
            T = song["length"]
            if T < self.chunk_size:
                # we're about to get an error if this is ever true. don't pad with zeros just let it fail
                if debug:
                    print(
                        f"\n__getitem__: songs[{idx}] = ({self.songs_listed[idx]}),  data.shape ={data.shape}, chunk_size = {self.chunk_size}",
                        flush=True,
                    )
            start = torch.randint(0, T - self.chunk_size, (1,))
            end = start + self.chunk_size
            out = data[:, start:end, :]
            if self.debug:
                print("\n__getitem__: out.shape = ", out.shape)
            yield out


class StemChunkStream(IterableDataset):
    """
    Infinite stream of random audio chunks from MUSDB18 stems using on-demand loading.

    Each sample is read directly from disk with `start` and `duration` to avoid loading
    full songs into memory.
    """

    def __init__(
        self,
        subset="train",
        data_dir="/home/shawley/datasets/musdb18-stems",
        chunk_size=2**18,  # number of samples per chunk
        sample_rate=44100,
        load_frac=1.0,  # fraction of dataset to use
        augment=False,
        debug=False,
    ):
        if debug:
            breakpoint()
        search_dir = os.path.join(data_dir, subset)
        self.songs_listed = sorted(glob(f"{search_dir}/*.mp4"))
        if load_frac < 1.0:
            keep_n = int(len(self.songs_listed) * load_frac)
            keep_n = max(keep_n, 1)
            self.songs_listed = random.sample(self.songs_listed, keep_n)

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.augment = augment
        self.debug = debug

        # Precompute song lengths in samples
        self.song_lengths = []
        for song_path in self.songs_listed:
            info = stempeg.Info(song_path)
            duration_sec = info.duration(0)  # dot notation, not ['duration']
            num_samples = int(duration_sec * info.sample_rate(0))
            self.song_lengths.append(num_samples)

        if debug:
            print(f"{subset}: {len(self.songs_listed)} songs listed.")
            print(
                f"Chunk size: {chunk_size} samples ({chunk_size / sample_rate:.2f} sec)"
            )
            print(f"{len(self.songs_listed)=}")

    def __iter__(self):
        if self.debug is True:
            print(f"{len(self.songs_listed)=}")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            seed = random.randint(0, 2**32 - 1)
        else:
            # Unique seed per worker
            seed = worker_info.seed

        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        rng = random.Random(seed)

        while True:  # Infinite stream
            # Pick a random song
            song_idx = rng.randint(0, len(self.songs_listed) - 1)
            song_path = self.songs_listed[song_idx]
            song_len = self.song_lengths[song_idx]

            if song_len <= self.chunk_size:
                continue  # skip too-short songs

            # Pick a random start position in seconds
            start_sample = rng.randint(0, song_len - self.chunk_size)
            start_sec = start_sample / self.sample_rate
            duration_sec = self.chunk_size / self.sample_rate

            # Load only the requested chunk
            try:
                data, _ = stempeg.read_stems(
                    song_path,
                    start=start_sec,
                    duration=duration_sec,
                    sample_rate=self.sample_rate,
                )
            except Exception as e:
                print(f"Worker {worker_info.id} failed on {song_path}: {e}")
                raise

            # Convert to torch tensor
            # data = torch.tensor(data, dtype=torch.float32)
            data = torch.from_numpy(data).float()

            if self.debug:
                print(
                    f"Loaded {song_path} [{start_sec:.2f}s -> {start_sec + duration_sec:.2f}s], shape={data.shape}"
                )

            if self.augment:
                data = augment_effects(data, self.sample_rate)

            # print(data.shape)
            yield data


class StemDataset2(Dataset):
    """This reads MUSDB18 stems files in .mp4 format. The contents of these are given in MUSDB18 docs:
    0 - The mixture,    <--- note we're not going to use this
    1 - The drums,
    2 - The bass,
    3 - The rest of the accompaniment,
    4 - The vocals.
    """

    def __init__(
        self,
        subset="train",  # 'train' or 'test'
        data_dir="/home/shawley/datasets/musdb18-stems",  # dir to look for songs
        preload=False,  # load all audio files into memory at init. If False, load on demand
        share_mem=False,  # share audio data memory between workers
        chunk_size=2**18,  # size of audio chunks to return
        sample_rate=44100,  # sample rate of audio
        load_frac=1.0,  # fraction of dataset to load
        debug=False,  # print debug info
    ):
        # breakpoint()
        search_dir = f"{data_dir}/{subset}"
        self.songs_listed = sorted(glob(f"{search_dir}/*.mp4"))
        print(f"{subset}: {len(self.songs_listed)} songs listed.  preload={preload}")
        self.songs = [None] * len(self.songs_listed)  # actual song data loaded
        self.songs_len = [None] * len(self.songs_listed)

        # automatically adjust chunk_size to sample rate vs 44100
        # if sample_rate != 44100:  chunk_size = int(chunk_size * sample_rate/44100)
        if debug:
            print("chunk_size = ", chunk_size)

        self.subset, self.chunk_size, self.sample_rate, self.debug = (
            subset,
            chunk_size,
            sample_rate,
            debug,
        )
        self.share_mem = share_mem
        self.load_count = 0
        self.song_data = None  #  this will be a shared array to store (zero-padded) song audio, persistently shared between all workers

    def load_song(self, idx, start=0, duration=None, debug=False):
        "loads one song file"
        if type(idx) is int:
            song_file = self.songs_listed[idx]
        elif type(idx) is str:
            song_file = idx
        else:
            print("Unsupported datatype = ", type(idx))
        self.load_count += 1  # note this doesn't really work with parallel loading, i.e. when num_workers>0 :-(
        if debug or self.debug:
            print(f"{self.subset}: Loading {song_file}", flush=True)
        data, sample_rate = stempeg.read_stems(
            song_file, sample_rate=self.sample_rate, start=start, duration=duration
        )
        data = torch.tensor(data, dtype=torch.float32)
        if debug:
            print(
                f"load_song {idx}: {self.songs_listed[idx]}: data.shape = ", data.shape
            )
        song_dict = {
            "name": song_file,
            "data": data,
            "sample_rate": sample_rate,
            "length": data.shape[1],
        }
        return song_dict

    def __len__(self):
        return (
            len(self.songs) * 100000
        )  # we're going to be grabbing random windows so...keep the party going

    def __getitem__(self, idx, debug=False):
        """Returns a random chunk of audio / grouped-stems from a random song"""
        # breakpoint()
        idx = torch.randint(
            0, len(self.songs), (1,)
        ).item()  # ignore the input idx, pick a random song
        # data = self.song_data[idx]  # self.songs[idx]['data']
        # breakpoint()
        song = self.load_song(idx, debug)
        data = song["data"]
        # T = self.songs[idx]["length"]  # the real length of the song
        T = song["length"]
        if T < self.chunk_size:
            # we're about to get an error if this is ever true. don't pad with zeros just let it fail
            if debug:
                print(
                    f"\n__getitem__: songs[{idx}] = ({self.songs_listed[idx]}),  data.shape ={data.shape}, chunk_size = {self.chunk_size}",
                    flush=True,
                )
        start = torch.randint(0, T - self.chunk_size, (1,))
        end = start + self.chunk_size
        out = data[:, start:end, :]
        if debug:
            print("\n__getitem__: out.shape = ", out.shape)
        return out.to(torch.float32)  # .to just to make sure...


class StemDataset(Dataset):
    """This reads MUSDB18 stems files in .mp4 format. The contents of these are given in MUSDB18 docs:
    0 - The mixture,    <--- note we're not going to use this
    1 - The drums,
    2 - The bass,
    3 - The rest of the accompaniment,
    4 - The vocals.
    """

    def __init__(
        self,
        subset="train",  # 'train' or 'test'
        data_dir="/home/shawley/datasets/musdb18-stems",  # dir to look for songs
        preload=True,  # load all audio files into memory at init. If False, load on demand
        share_mem=True,  # share audio data memory between workers
        chunk_size=2**18,  # size of audio chunks to return
        sample_rate=44100,  # sample rate of audio
        load_frac=1.0,  # fraction of dataset to load
        debug=False,  # print debug info
    ):
        # breakpoint()
        search_dir = f"{data_dir}/{subset}"
        self.songs_listed = sorted(glob(f"{search_dir}/*.mp4"))
        print(f"{subset}: {len(self.songs_listed)} songs listed.  preload={preload}")
        self.songs = [None] * len(self.songs_listed)  # actual song data loaded

        # automatically adjust chunk_size to sample rate vs 44100
        # if sample_rate != 44100:  chunk_size = int(chunk_size * sample_rate/44100)
        if debug:
            print("chunk_size = ", chunk_size)

        self.subset, self.chunk_size, self.sample_rate, self.debug = (
            subset,
            chunk_size,
            sample_rate,
            debug,
        )
        self.share_mem = share_mem
        self.load_count = 0
        self.song_data = None  #  this will be a shared array to store (zero-padded) song audio, persistently shared between all workers
        if preload:
            self.preload(load_frac=load_frac)

    def load_song(self, idx, debug=False):
        "loads one song file"
        if type(idx) is int:
            song_file = self.songs_listed[idx]
        elif type(idx) is str:
            song_file = idx
        else:
            print("Unsupported datatype = ", type(idx))
        self.load_count += 1  # note this doesn't really work with parallel loading, i.e. when num_workers>0 :-(
        if debug or self.debug:
            print(f"{self.subset}: Loading {song_file}", flush=True)
        data, sample_rate = stempeg.read_stems(song_file, sample_rate=self.sample_rate)
        data = torch.tensor(data, dtype=torch.float32)
        if debug:
            print(
                f"load_song {idx}: {self.songs_listed[idx]}: data.shape = ", data.shape
            )
        song_dict = {
            "name": song_file,
            "data": data,
            "sample_rate": sample_rate,
            "length": data.shape[1],
        }
        return song_dict

    def group_audio_data(self):
        """creates a a big (shared memory?) array  for audio data (one that's common to all workers)"""
        n_songs, n_stems, n_channels = len(self.songs), 5, 2
        max_len = 0  # we need to find out the longest song (in samples) to fit in the data array
        for song in self.songs:
            if song is not None:
                max_len = max(max_len, song["length"])
        if self.share_mem:
            print("     Creating shared memory array...")
            shared_array_base = mp.Array(
                ctypes.c_float, n_songs * n_stems * max_len * n_channels
            )
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            shared_array = shared_array.reshape(n_songs, n_stems, max_len, n_channels)
            shared_array = np.zeros((n_songs, n_stems, max_len, n_channels))
            self.song_data = torch.from_numpy(shared_array)
        else:  # to compare against non-sharing way of doing things
            self.song_data = torch.zeros((n_songs, n_stems, max_len, n_channels))
        for i, song in enumerate(self.songs):
            self.song_data[i, :, : song["length"], :] = song[
                "data"
            ]  # copy the data over to the shared array
        if (
            self.share_mem
        ):  # here's the key: now remove the non-shared audio data from memory!
            for song in self.songs:
                song.pop("data")

    def preload(self, num_workers=min(12, os.cpu_count()), load_frac=1.0, debug=False):
        """Preloads all songs - in parallel. May not be feasible for large datasets."""
        print(f"{self.subset}: Preloading songs...")
        self.songs = []
        max_ = max(1, int(len(self.songs_listed) * load_frac))
        num_workers = 1
        if num_workers > 1:  # parallel loading, fast but often hangs
            with mp.Pool(processes=num_workers) as p:
                with tqdm(total=max_) as pbar:
                    for r in p.imap_unordered(self.load_song, range(0, max_)):
                        self.songs.append(r)
                        pbar.update()
        else:  # sequential is slow but shure
            for i in trange(max_):
                self.songs.append(self.load_song(i))
        # just to be sure... rewrite the song list (ordering) based on what we got back from the read:
        self.songs_listed = [x["name"] for x in self.songs]
        self.group_audio_data()

    def __len__(self):
        return (
            len(self.songs) * 100000
        )  # we're going to be grabbing random windows so...keep the party going

    def __getitem__(self, idx, debug=True):
        """Returns a random chunk of audio / grouped-stems from a random song"""
        idx = torch.randint(
            0, len(self.songs), (1,)
        ).item()  # ignore the input idx, pick a random song
        data = self.song_data[idx]  # self.songs[idx]['data']
        T = self.songs[idx]["length"]  # the real length of the song
        if T < self.chunk_size:
            # we're about to get an error if this is ever true. don't pad with zeros just let it fail
            if debug:
                print(
                    f"\n__getitem__: songs[{idx}] = ({self.songs_listed[idx]}),  data.shape ={data.shape}, chunk_size = {self.chunk_size}",
                    flush=True,
                )
        start = torch.randint(0, T - self.chunk_size, (1,))
        end = start + self.chunk_size
        out = data[:, start:end, :]
        if debug:
            print("\n__getitem__: out.shape = ", out.shape)
        return out.to(torch.float32)  # .to just to make sure...


class EncodingsDataset(Dataset):
    """This reads precomputed encodings from disk.
    The encodings are assumed to be in the same order for each part.
    """

    def __init__(
        self,
        subset="train",  # 'train' or 'test'
        data_dir="/data/05-03_VGGish_1min_Encodings",  # dir to look for songs
        preload=True,  # load all audio files into memory at init. If False, load on demand
        chunk_size=590,  # size of windows of encoding-spectrograms chunks to return
        debug=False,  # print debug info
        ext=".pt",
    ):
        # check if '/train' and '/test' dirs exist in data_dir
        if not os.path.isdir(f"{data_dir}/train") or not os.path.isdir(
            f"{data_dir}/test"
        ):
            print("Taking a moment to build train/ and test/ in data_dir...")
            build_vggish_stemlike(data_dir)

        search_dir = f"{data_dir}/{subset}"
        print("Searching in", search_dir)
        self.songs_listed = sorted(glob(f"{search_dir}/*{ext}"))
        print(f"{subset}: {len(self.songs_listed)} songs listed.  preload={preload}")
        self.songs = [None] * len(self.songs_listed)  # actual song data loaded
        self.subset, self.chunk_size, self.debug = subset, chunk_size, debug
        self.load_count = 0
        if preload:
            self.preload()

    def load_song(self, idx, debug=False):
        "appends song data self.songs"
        if type(idx) is int:
            song_file = self.songs_listed[idx]
        elif type(idx) is str:
            song_file = idx
        else:
            print("Unsupported datatype = ", type(idx))
        self.load_count += 1  # note this doesn't really work with parallel loading, i.e. when num_workers>0 :-(
        if debug or self.debug:
            print(f"{self.subset}: Loading {song_file}", flush=True)
        data, sample_rate = torch.load(song_file), 44100
        # data = torch.tensor(data, dtype=torch.float32)
        song_dict = {"name": song_file, "data": data, "sample_rate": sample_rate}
        # self.songs[idx] = song_dict  # not great for parallel loading
        return song_dict

    def preload(self, debug=False):
        """Preloads all songs - in parallel. May not be feasible for large datasets."""
        print(f"{self.subset}: Preloading songs...")
        self.songs = []
        max_ = len(self.songs_listed)
        with mp.Pool(
            processes=mp.cpu_count() // 8
        ) as p:  # the //8 is just so we get to see the prog bar doing something! ;-)
            with tqdm(total=max_) as pbar:
                for r in p.imap_unordered(self.load_song, range(0, max_)):
                    self.songs.append(r)
                    pbar.update()
        # just to be sure... rewrite the song list (ordering) based on what we got back from the parallel read:
        self.songs_listed = [
            x["name"] for x in self.songs
        ]  # TODO: this should really go the other way

    def __len__(self):
        if self.subset == "test":  # don't want validation to go on forever
            return len(self.songs) * 10
        else:
            return (
                len(self.songs) * 100000
            )  # we're going to be grabbing random windows so...keep the party going

    def __getitem__(self, idx, debug=False):
        # ignore the input idx, pick a random song
        idx = torch.randint(0, len(self.songs), (1,)).item()
        if debug or self.debug:
            print(f"idx = {idx}, len(self.songs) {len(self.songs)}")
        if self.songs[idx] is None:
            self.songs[idx] = self.load_song(idx)
        data = self.songs[idx]["data"]
        S, T, C = data.shape  # batch, stems, time, channels
        start = torch.randint(0, T - self.chunk_size, (1,))
        end = start + self.chunk_size
        return data[:, start:end, :]


class MTGJamendoLazy(Dataset):
    """Dataset that lazily loads and caches audio files.

    On first access, the full audio file is loaded, resampled to self.sample_rate,
    cached in memory, and then random chunking is applied.
    """

    def __init__(
        self,
        data_dir="/scratch/users/nshaheed/mtg-jamendo",
        chunk_size=2**18,
        sample_rate=44100,
        load_frac=1.0,
        load_count=None,  # can ony do on or the other
        augment=False,
        debug=False,
    ):
        self.songs_listed = sorted(glob(f"{data_dir}/*/*.wav"))
        self.chunk_size = chunk_size
        self.debug = debug
        self.augment = augment
        self.sample_rate = sample_rate

        # Fractional subset of files
        if load_count is not None:
            keep_n = max(load_count, 1)
        if load_frac < 1.0:
            keep_n = int(len(self.songs_listed) * load_frac)
            keep_n = max(keep_n, 1)

        self.songs_listed = random.sample(self.songs_listed, keep_n)

        self.audio_cache = {}  # cache for loaded + resampled audio arrays

    def __len__(self):
        return len(self.songs_listed)

    def __getitem__(self, idx):
        if self.debug:
            print(f"{len(self.songs_listed)=}")

        # worker-seeded randomness
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.seed if worker_info else random.randint(0, 2**32 - 1)
        rng = random.Random(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)

        # breakpoint()
        song_path = self.songs_listed[idx]

        # Lazily load + resample if not already cached
        if song_path not in self.audio_cache:
            print(f"caching idx {idx}, {song_path}...")
            wv, orig_sr = sf.read(song_path, dtype="float32", always_2d=True)
            wv = torch.from_numpy(wv)

            wv = wv[:, 0]  # make mono

            if orig_sr != self.sample_rate:
                wv = torchaudio.functional.resample(wv, orig_sr, self.sample_rate)

            self.audio_cache[song_path] = wv

        wv = self.audio_cache[song_path]
        length = wv.shape[0]

        # Pick a random starting point
        if length <= self.chunk_size:
            start = 0
        else:
            start = torch.randint(length - self.chunk_size, size=(1,)).item()

        chunk = wv[start : start + self.chunk_size]

        return chunk


# --- utility routine:


def build_vggish_stemlike(data_dir="/data/05-03_VGGish_1min_Encodings"):
    in_subsets = [x + "_VGGish" for x in ["Train", "Test"]]
    for in_s in in_subsets:
        assert os.path.isdir(f"{data_dir}/{in_s}"), f"{data_dir}/{in_s} does not exist"
    out_subsets = ["train", "test"]
    """
        0 - The mixture,
        1 - The drums,
        2 - The bass,
        3 - The rest of the accompaniment,
        4 - The vocals.
    """
    parts = ["mix", "drums", "bass", "other", "vocals"]
    for out_s in out_subsets:
        os.makedirs(f"{data_dir}/{out_s}", exist_ok=True)
    for in_subset in in_subsets:
        # get a list of input mix files
        search_str = f"{data_dir}/{in_subset}/{parts[0]}/*.pt"
        in_mix_files = glob(search_str)
        print(f"Searching in {search_str} found {len(in_mix_files)} songs")
        for in_mix in in_mix_files:
            print("in_mix =", in_mix)
            mix = torch.tensor(torch.load(in_mix))
            encoding_stems = torch.empty((5, mix.shape[0], mix.shape[1]))
            encoding_stems[0] = mix
            for i, p in enumerate(parts[1:]):
                in_file = in_mix.replace("mix", p)
                in_stem = torch.tensor(torch.load(in_file))
                encoding_stems[1 + i] = in_stem
            out_subset = "train" if "train" in in_subset.lower() else "test"
            out_file = in_mix.replace("/mix", "").replace(in_subset, out_subset)
            print("    Saving to", out_file)
            torch.save(encoding_stems, out_file)

        # print(in_files)


if __name__ == "__main__":
    import sys

    # test MTGJamendo
    mtg = MTGJamendo()
    next(iter(mtg))

    # test MTGJamendoStream
    # mtg = MTGJamendoStream()
    # stemcnk = StemChunkStream(data_dir="/scratch/users/nshaheed/musdb18")
    # chain = ChainDataset([mtg, stemcnk])

    # result = next(iter(mtg))
    # result_cnk = next(iter(stemcnk))
    # result1 = next(iter(chain))
    # result2 = next(iter(chain))
    # breakpoint()

    # only need to run the following once:
    # build_vggish_stemlike()
    # sys.exit(0)

    # test the dataset
    # test_ds = EncodingsDataset(subset="test", preload=True, debug=False)
    # ds_iter = iter(test_ds)
    # songs = []
    # songs.append(next(ds_iter))
    # songs.append(next(ds_iter))
    # for s, song in enumerate(songs):
    #     print(f"songs[{s}].shape = ", song.shape)
    #     for i in range(song.shape[0]):
    #         for j in range(song.shape[1]):
    #             vec = song[i, j, :]
    #             vec_norm = vec.norm()
    #             Print(f"song[{s}][{i},{j},:].norm() = {vec_norm:.3f}")
