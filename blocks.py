from pathlib import Path
from tqdm import tqdm
import random
import math

import librosa
import wandb
from oplas.models import Music2Latent, Projector, ProjectorVAE, VAE

from music2latent import EncoderDecoder

import torch
import torchaudio
import numpy as np

from latent_effects import Gainn
from helper import get_projector, get_gain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

projector = get_projector(device=device)
gain_effect = get_gain(device=device)

encdec = EncoderDecoder()

# our stems
# file2='./stems/harmony2_pad/Alto pt. 1_1.wav'
file = "./stems/harmony2_pad/Alto pt. 2 (double)_1.wav"
file = "./kid_a.wav"

wv, _ = librosa.load(file, sr=44100)
wv2 = wv[590150 + 1024 * 8 : 44100 * 14 + 1024 * 8]
wv = 1.0 * wv[590150 : 44100 * 14]

# wv2, _ = librosa.load(file2, sr=44100)
# wv2 = 1.0 * wv2[:44100*45]

latent = encdec.encode(wv).squeeze()
latent2 = encdec.encode(wv2).squeeze()
silence = encdec.encode(np.zeros_like(wv)).squeeze()

distance = latent2 - latent


# points = [
#     latent[:,200],
#     latent[:,204],
#     latent[:,208],
# ]

# point_towards = latent2[:,-2]

points = [
    # latent[:,148],
    # point_towards,
    # latent[:,152],
    # latent[:,154],
]


# rollers = []

length = 80

# for i in range(length):
#     # p = point.roll(i)
#     # rollers.append(p)

# latent = torch.stack(rollers).permute(1,0)
# breakpoint()


loop = latent.shape[-1]
base = latent.repeat(1, length)
latent1 = latent.repeat(1, length)  # some beatz
latent2 = latent2.repeat(1, length)  # some beatz
latent3 = latent.repeat(1, length)  # some beatz
diff = latent2 - latent1
# breakpoint()
silence = silence.repeat(1, length)

# slab out a sub-tensor
block = "big"
# latent[:block,:] = silence[:block,:]

# breakpoint()


for i in range(length * loop):
    offset = math.floor(i / 6) % 64

    # latent[offset:offset+block,i] = silence[offset:offset+block,i]
    # latent[:,i] = latent[:,i] + 0.01 * i
    latent1[:, i] = latent1[:, i] + 0.0009 * float(i) * diff[:, i]

    bias3 = 0.0007 * float(i) * diff[:, i]
    latent3[:, i] = latent3[:, i] + bias3


latent1 = gain_effect(latent1, 0.25)  # will this work??
latent3 = gain_effect(latent3, 0.25)  # will this work??

latents = [
    base,
    latent1,
    latent3,
]

# point1_rpt = point.repeat(length,1).permute(1,0)

# for point in points:
#     points = point.repeat(length,1).permute(1,0)
#     for i in range(length):
#         swaps = 1

#         # if i % 50 == 0 or i % 50 == 1:
#         #     swaps = 30

#         scale = i % 50

#         # should increase density over time
#         if i > 0 and scale < 3*math.log2(i):
#             swaps = int(4*math.log2(i))
#             swaps = int(60 * (float(i) / length)**2)
#             swaps = min(swaps, 40)
#             print(f'{swaps=}')


#         for l in range(swaps):
#             # l, r = random.randint(0, 63), random.randint(0, 63)

#             #randomly swap dims
#             # points[[l,r], i] = points[[r,l], i]
#             # points[l,i] = point_towards[l]
#             points[l,i] = latent2[l,scale]


#     latents.append(points)


# for point1, point2 in zip(points, points2):
#     length = 600
#     point1_rpt = point1.repeat(length,1).permute(1,0)
#     point2_rpt = point2.repeat(length,1).permute(1,0)
#     ramp = torch.linspace(0, 1, steps=length)
#     ramp_rev = torch.linspace(1, 0, steps=length)
#     rands = ramp * torch.rand_like(point1_rpt)
#     rands_rev = ramp_rev * torch.rand_like(point2_rpt)
#     point_rpt = (1 - ramp) * (point1_rpt + rands) + ramp * (point2_rpt + rands_rev)
#     latents.append(point_rpt)

# latents.append(latent)

# ramp = torch.linspace(0, 2, steps=latent.shape[-1])
# # ramp = torch.clamp(ramp, min=0, max=1)
# rands = ramp * torch.rand_like(ramp)

# # breakpoint()
# lerp = rands * latent2 + (1-rands) * latent

# ramp2 = torch.linspace(0, 1.5, steps=latent.shape[-1])
# # ramp2 = torch.clamp(ramp2, min=0, max=1)
# rands2 = ramp2 * torch.rand_like(ramp2)

# # lerp = rands * latent2 + (1-rands) * latent
# lerp2 = rands2 * latent2 + (1-rands2) * latent

# latents.append(lerp)
# latents.append(lerp2)


stack = torch.stack(latents).permute(0, 2, 1)
# breakpoint()

print("rendering audio...")
z_stack = projector.encode(stack)

z_sum = z_stack.sum(0)

latent_sum = projector.decode(z_sum).permute(1, 0)

if len(latents) == 1 and False:
    audio = encdec.decode(latents[0])
else:
    audio = encdec.decode(latent_sum)

# encode into projections space

print(f"writing blocks-{block}.wav")
torchaudio.save(f"blocks-{block}.wav", audio, 44100)
