import torch
import soundfile as sf
import torchaudio
from multiprocessing import Pool
import os, math, glob, random

from tqdm import tqdm


def load_audio(path, sample_rate=16000):
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


# Top-level wrapper so it's picklable
def _load_audio_wrapper(args):
    path, sample_rate = args
    return load_audio(path, sample_rate)


def multiprocess_load_sharded(
    file_list,
    sample_rate=16000,
    num_workers=8,
    cache_dir="cache",
    shard_size=1000,
    fraction=1.0,
    shuffle=True,
):
    """
    Load audio dataset with caching in shards.
    - First run: load in parallel, save shards to cache_dir.
    - Later runs: load shards from cache_dir.
    - fraction < 1.0 loads only a fraction of shards (random or first few).
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Check for existing shards
    shard_paths = sorted(glob.glob(os.path.join(cache_dir, "cache_*.pt")))
    # if shard_paths:
    #     print(f"Found {len(shard_paths)} shards in {cache_dir}.")

    #     # Pick fraction of shards
    #     n_shards = max(1, int(len(shard_paths) * fraction))
    #     if shuffle:
    #         shard_paths = random.sample(shard_paths, n_shards)
    #     else:
    #         shard_paths = shard_paths[:n_shards]

    #     print(f"Loading {len(shard_paths)} shard(s) from cache...")
    #     tensors = []
    #     for p in shard_paths:
    #         tensors.extend(torch.load(p))
    #     return tensors

    # if shards already exist, then skip ahead
    shard_idx = len(shard_paths)
    file_list = file_list[shard_size * shard_idx :]

    # No cache yet: load in parallel
    print("Cache not found, loading audio files in parallel...")
    args = [(f, sample_rate) for f in file_list]
    # with Pool(processes=num_workers) as pool:
    #     tensors = pool.starmap(load_audio, args)

    # with Pool(processes=num_workers) as pool:
    #     tensors = list(tqdm(pool.imap_unordered(_load_audio_wrapper, args),
    #                         total=len(file_list), desc="Decoding audio"))

    batch = []
    # shard_idx += 1
    with Pool(processes=num_workers) as pool:
        for tensor in tqdm(
            pool.imap_unordered(_load_audio_wrapper, args),
            total=len(file_list),
            desc="Decoding audio",
        ):
            batch.append(tensor)
            # When batch reaches shard_size, save and free memory
            if len(batch) >= shard_size:
                shard_path = os.path.join(cache_dir, f"cache_{shard_idx}.pt")
                torch.save(batch, shard_path)
                print(f"Saved shard {shard_idx} ({len(batch)} samples) -> {shard_path}")
                batch = []  # free memory
                shard_idx += 1

        # Save any remaining files
        if batch:
            shard_path = os.path.join(cache_dir, f"cache_{shard_idx}.pt")
            torch.save(batch, shard_path)
            print(
                f"Saved final shard {shard_idx} ({len(batch)} samples) -> {shard_path}"
            )

    print(f"All shards written to {cache_dir}")


# Example usage
if __name__ == "__main__":
    files = sorted(glob.glob("/scratch/users/nshaheed/mtg-jamendo-wav/*/*.wav"))

    # files = files[:100]
    cache_dir = "/scratch/users/nshaheed/mtg-jamendo-cache"
    sample_rate = 44100

    # First run: builds shards
    multiprocess_load_sharded(
        files,
        sample_rate=sample_rate,
        num_workers=12,
        cache_dir=cache_dir,
        shard_size=16,
    )
    # print(f"First run loaded {len(tensors)} tensors")

    # Later run: load only 50% of shards
    # tensors = multiprocess_load_sharded(
    #     files, sample_rate=sample_rate, num_workers=4, cache_dir=cache_dir, fraction=0.5
    # )
    # breakpoint()
    # print(f"Second run loaded {len(tensors)} tensors (fraction=0.5)")
