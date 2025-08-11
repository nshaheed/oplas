import argparse
import time

import stempeg
import torch
import torchaudio

from oplas.mixing import mix_and_encode
from oplas.models import Music2Latent, Projector

parser = argparse.ArgumentParser(description="Generate audio for quick tests.")
parser.add_argument("checkpoint")

parser.add_argument(
    "-d", "--device", type=str, default="cpu", help="which device to run benchmark"
)

parser.add_argument("-t", "--track", type=int, help="specific track in encode/decode")

args = parser.parse_args()

device = torch.device(args.device)
track = args.track
projector = torch.load(
    args.checkpoint,
    map_location=device,
    weights_only=False,
)
# projector = Projector(64, 64, trivial=True)

song_file = "data/test/Al James - Schoolboy Facination.stem.mp4"
# song_file = "data/train/James May - All Souls Moon.stem.mp4"

data, sr = stempeg.read_stems(song_file, sample_rate=44100)
data = torch.tensor(data, dtype=torch.float32)
data = data[:, : 20 * sr, :]  # grab the first 4 seconds
data = data.unsqueeze(0)  # add our "batch" dimension

encoder = Music2Latent()
mixes = mix_and_encode(data, encoder)

# ys - a list[] of the 4 stems encded with music2latent
# y_mix - the full mix audio encoded into a single file
# g_stems - the individual stems audio files
# g_mix - the mixed audio file

# Things I want to do
# - decode y_mix and save file
# - project ys->zs, sum, and z->y, decode, save file
# - do this with full mix, single audio file

# save original snippet
if track:
    y_mix_decoded = encoder.decode(mixes["ys"][track - 1])
else:
    y_mix_decoded = encoder.decode(mixes["y_mix"])
# breakpoint()

if track:
    torchaudio.save("original.wav", data[0, track].permute(1, 0), sample_rate=sr)
else:
    torchaudio.save("original.wav", data[0, 0].permute(1, 0), sample_rate=sr)


torchaudio.save("y_mix.wav", y_mix_decoded, sample_rate=sr)

zs = []

if track:
    ys = [mixes["ys"][track - 1]]
else:
    ys = mixes["ys"]
for y in ys:
    if len(y.shape) == 2:  # handle single stem case
        y = y.unsqueeze(0)
    # breakpoint()
    y = y.permute(0, 2, 1)
    z = projector.encode(y)
    zs.append(z.permute(0, 2, 1))

z_sum = sum(zs)
z_sum = z_sum.permute(0, 2, 1)
print(f"{z_sum.shape=}")

# breakpoint()

z_mix = projector.decode(z_sum)
z_mix = z_mix.permute(0, 2, 1)

z_mix_decoded = encoder.decode(z_mix)
torchaudio.save("z_mix.wav", z_mix_decoded, sample_rate=sr)
# breakpoint()
