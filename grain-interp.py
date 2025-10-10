from latent_effects import Gain
from helper import get_projector, get_gain, plot_audio, plot_audio2, slerp

import librosa
import torch
import torchaudio

import random

from music2latent import EncoderDecoder

# seed = random.randint(0, 2**32 - 1)
seed = 42
# np.random.seed(seed % (2**32 - 1))
torch.manual_seed(seed)
rng = random.Random(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

projector = get_projector(artifact_key="oauvlx7z", version="v142", device=device) # old 4 voice
# projector = get_projector(artifact_key="00245wks", version="v73", device=device) # new 16 voice
gain_effect = get_gain(device=device)

encdec = EncoderDecoder()

file = "./kid_a.wav"
wv, _ = librosa.load(file, sr=44100)
wv = 0.25 * wv[590150 : 44100 * 14]

ending_wv, _ = librosa.load("./kid_a_ending.wav", sr=44100)

kick = encdec.encode(wv).squeeze().permute(1, 0)
ending = encdec.encode(ending_wv).squeeze().permute(1, 0)

# breakpoint()

# get files of stems of victoria's stack
stem_files = [
    "./stems/chord/chord-01-2.wav",
    "./stems/chord/chord-02-2.wav",
    "./stems/chord/chord-03-2.wav",
    "./stems/chord/chord-04-2.wav",
    "./stems/chord/chord-05-2.wav",
    "./stems/chord/chord-06-2.wav",
    "./stems/chord/chord-07-2.wav",
    "./stems/chord/chord-08-2.wav",
    "./stems/chord/chord-09-2.wav",
    "./stems/chord/chord-10-2.wav",
    "./stems/chord/chord-11-2.wav",
    "./stems/chord/chord-12-2.wav",
    "./stems/chord/chord-13-2.wav",
    "./stems/chord/chord-14-2.wav",
    "./stems/chord/chord-15-2.wav",
    "./stems/chord/chord-16-2.wav",
    "./stems/chord/chord-17-2.wav",
]

stems = []

# encode EVerything
for path in stem_files:
    wav, _ = librosa.load(path, sr=44100)
    wav = 0.5 * wav
    stems.append(encdec.encode(wav).squeeze().permute(1, 0))

length = 1000

z_sum = torch.zeros([length, 256])

# build up a base layer
i = 0
while i < length - 6:
    # idx = rng.randint(0, z.shape[0]-1)
    # dur = rng.randint(0, z.shape[0]-idx-1)
    idx = 0
    dur = 6

    result = kick[idx : idx + dur]
    dur = result.shape[0]

    if rng.random() < 0.3:
        flip = torch.flip(result, (0,))
        flip = projector.encode(flip)
        z_sum[i : i + dur, :] += flip

    z_sum[i : i + dur, :] += projector.encode(result)
    i += dur + 4


for k in range(9):
    beg = 40 + int(0.06 * (k + 1) * 600)

    for i in range(beg, length - 6):
        play_chance = 0.2
        if k > 7:
            play_chance = 0.3

        # if i > 600:
        #     play_chance = 0.4

        if rng.random() < 0.2:
            # print(f"{i}/{length}")
            idx_k = 0
            dur = 6
            dur = rng.randint(0, kick.shape[0] - idx_k - 1)

            result = kick[idx_k : idx_k + dur]
            dur = result.shape[0]

            odds = 0.5
            if k > 5:
                odds = 1.0

            if k >= 8:
                odds = 1.0

            if i > 600:
                odds = 1.0

            if rng.random() < odds:
                # do inerpolation
                scale = rng.random() * 2.0
                idx = rng.randint(0, len(stems) - 1)
                # dur = rng.randint(0, kick.shape[0]-idx_k-1)

                # result = (1 - scale) * result + scale * stems[idx][:dur]
                # print(f'{scale=}')
                new_result = torch.zeros_like(result)
                for j in range(dur):
                    frac = scale * (float(j) / dur) ** 0.5

                    new_result[j] = slerp(result[j], stems[idx][j], frac)

                    # result = (1 - scale) * result + scale * stems[idx][:dur]
                    # result[j] = (1-frac) * result[j] + frac * stems[idx][j]

                result = new_result
                # result = slerp(result, stems[idx][:dur], scale)
            elif rng.random() < 0.3:
                # add some NOISE
                # result = result + 0.5 * rng.random() * torch.rand_like(result)
                result = result + 0.5 * torch.rand_like(result)
            # result = result.permute(1,0)

            # print(f'z_sum[{i}:{i+dur},:]')
            # print(f'{projector.encode(result).shape=}')
            z_sum[i : i + dur, :] += projector.encode(result)
            # i += dur+4


latent_sum = projector.decode(z_sum).permute(1, 0)

# end it
for i in range(600, length):
    scale = 0.9 * (float(i - 600) / 300)
    scale = min(scale, 0.9)
    # print(f"{scale=}")

    # latent_sum[:,i] = latent_sum[:,i] + scale * latent_sum[:,i]

    # scale shows balance, random amount of movement between

    # # Random indices (one per row)
    # indices = torch.randint(0, ending.size(0), (ending.size(1), 1))

    # # Gather values at those indices
    # result = torch.gather(ending, 0, indices).squeeze()

    idx = rng.randint(0, ending.size(0) - 1)
    result = ending[idx, :].squeeze()

    new_point = slerp(latent_sum[:, i], result, scale)
    latent_sum[:, i] = new_point

    # latent_sum[:,i] = latent_sum[:,i] + scale * torch.rand_like(latent_sum[:,i])


audio = encdec.decode(latent_sum)

torchaudio.save(f"grain-interp.wav", audio, 44100)
