from latent_effects import Gain
from helper import get_projector, get_gain, plot_audio, plot_audio2

import librosa
import torch
import torchaudio

import random

from music2latent import EncoderDecoder

random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

projector = get_projector(artifact_key="oauvlx7z", version="v142", device=device)
# projector = get_projector(device=device)
gain_effect = get_gain(device=device)

encdec = EncoderDecoder()

file = "./kid_a.wav"
wv, _ = librosa.load(file, sr=44100)
wv = 0.25 * wv[590150 : 44100 * 14]


# plot_audio(wv)

# one latent is 4352 samples

latent = encdec.encode(wv).squeeze()

# latent = encdec.encode(wv)
# new_audio = encdec.decode(latent).squeeze()
# plot_audio2(wv, new_audio)
# breakpoint()

# proj_space
z = projector.encode(latent.permute(1, 0))


length = 600

z_sum = torch.zeros([length, 256])

# build up a base layer
i = 0
while i < length - 6:
    # idx = random.randint(0, z.shape[0]-1)
    # dur = random.randint(0, z.shape[0]-idx-1)
    idx = 0
    dur = 6

    result = z[idx : idx + dur]
    dur = result.shape[0]

    if random.random() < 0.3:
        flip = torch.flip(result, (0,))
        z_sum[i : i + dur, :] += flip

    z_sum[i : i + dur, :] += result
    i += dur + 4


# randomly insert in projector space
for j in range(9):
    beg = int(0.1 * (j + 1) * length)
    for i in range(beg, length - 6):
        if random.random() < 0.2:
            if j > 1:
                idx = random.randint(0, z.shape[0] - 4)
            else:
                idx = random.randint(0, z.shape[0] - 1)
            # breakpoint()
            dur = random.randint(0, z.shape[0] - idx - 1)

            # idx = 0
            # dur = 6

            # print(f'{idx=}, {dur=}')
            result = z[idx : idx + dur]
            dur = result.shape[0]

            # breakpoint()

            if random.random() < 0.3:
                result = torch.flip(result, (0,))

            z_sum[i : i + dur, :] += result


latent_sum = projector.decode(z_sum).permute(1, 0)
audio = encdec.decode(latent_sum)

torchaudio.save(f"grain-hit-new.wav", audio, 44100)
