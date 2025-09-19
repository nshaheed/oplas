import os
import time
import soundfile as sf
from tqdm import tqdm

def load_audio_files(directory, max_duration=30.0):
    """Load the first `max_duration` seconds of audio files in a directory using soundfile."""
    audio_data = {}
    start_time = time.perf_counter()

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    for filename in tqdm(files, desc="Loading audio files"):
        filepath = os.path.join(directory, filename)

        try:
            with sf.SoundFile(filepath) as f:
                samplerate = f.samplerate
                frames_to_read = int(max_duration * samplerate)
                data = f.read(frames=frames_to_read, dtype="float32", always_2d=True)
                audio_data[filename] = (data, samplerate)
        except RuntimeError as e:
            tqdm.write(f"Skipping {filename}: {e}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"\nLoaded {len(audio_data)} audio files (up to {max_duration}s each) in {elapsed_time:.3f} seconds")
    return audio_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark loading audio files with soundfile (first N seconds).")
    parser.add_argument("directory", help="Path to directory containing audio files")
    parser.add_argument("--duration", type=float, default=30.0, help="Max duration (seconds) to load per file")
    args = parser.parse_args()

    load_audio_files(args.directory, max_duration=args.duration)
