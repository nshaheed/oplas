import os
import time
import soundfile as sf
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob


def load_file(filepath, max_duration):
    """Helper to load up to `max_duration` seconds of audio from a single file."""
    try:
        with sf.SoundFile(filepath) as f:
            samplerate = f.samplerate
            frames_to_read = int(max_duration * samplerate)
            data = f.read(frames=frames_to_read, dtype="float32", always_2d=True)
            # Estimate memory footprint in bytes
            size_bytes = data.nbytes
            # return filepath, (data, samplerate, size_bytes)
            return filepath, (samplerate, size_bytes)

    except Exception as e:
        return filepath, e


def load_audio_files_parallel(directory, max_duration=30.0, num_workers=None):
    """Load the first `max_duration` seconds of audio files in parallel."""
    audio_data = {}
    total_bytes = 0

    # files = [
    #     os.path.join(directory, f)
    #     for f in os.listdir(directory)
    #     if os.path.isfile(os.path.join(directory, f))
    # ]

    files = sorted(glob(f"{directory}/*/*.mp3"))

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_file, f, max_duration): f for f in files}

        for future in tqdm(
            as_completed(futures), total=len(files), desc="Loading audio files"
        ):
            filepath, result = future.result()
            filename = os.path.basename(filepath)
            if isinstance(result, Exception):
                tqdm.write(f"Skipping {filename}: {result}")
            else:
                samplerate, size_bytes = result
                # audio_data[filename] = (data, samplerate)
                # audio_data[filename] = filename
                total_bytes += size_bytes

    elapsed_time = time.perf_counter() - start_time

    # Convert throughput to MB/sec
    throughput_mb = (
        (total_bytes / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0.0
    )

    print(
        f"\nLoaded {len(audio_data)} audio files "
        f"(up to {max_duration}s each) in {elapsed_time:.3f} seconds"
    )
    print(f"Total decoded audio: {total_bytes / (1024 * 1024):.2f} MB")
    print(f"Throughput: {throughput_mb:.2f} MB/sec")

    return audio_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark parallel audio loading with soundfile."
    )
    parser.add_argument("directory", help="Path to directory containing audio files")
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Max duration (seconds) to load per file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    args = parser.parse_args()

    load_audio_files_parallel(
        args.directory, max_duration=args.duration, num_workers=args.workers
    )
