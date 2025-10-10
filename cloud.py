from pathlib import Path
from tqdm import tqdm

import librosa
import wandb
from oplas.models import Music2Latent, Projector, ProjectorVAE, VAE

from music2latent import EncoderDecoder

import torch
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id = "uoznl0zb"  # fallen-sweep-4

projector = Projector(in_dims=64, out_dims=256, num_inner_layers=1, hidden_dims_scale=8)

# run = wandb.init(project="oplas", job_type="inference")
api = wandb.Api()
try:
    artifact = api.artifact(
        f"nshaheed-stanford-university/oplas/model-ckpt-{id}:latest"
    )
    artifact_dir = artifact.download()
    checkpoint_path = Path(artifact_dir) / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    projector.load_state_dict(checkpoint["model_state_dict"])
except Exception as e:
    print(f"Could not resume. Starting from scratch. Error: {e}")


encdec = EncoderDecoder()

# our stems
# file2='./stems/harmony2_pad/Alto pt. 1_1.wav'
file2 = "./stems/harmony2_pad/Alto pt. 2 (double)_1.wav"
file = "./kid_a.wav"

wv, _ = librosa.load(file, sr=44100)
wv = 1.0 * wv[: 44100 * 33]

wv2, _ = librosa.load(file2, sr=44100)
wv2 = 1.0 * wv2[: 44100 * 33]

latent = encdec.encode(wv).squeeze()
latent2 = encdec.encode(wv2).squeeze()

points = [
    latent[:, 200],
    latent[:, 204],
    latent[:, 208],
]

points2 = [
    latent[:, 148],
    latent[:, 152],
    latent[:, 154],
]

latents = []

for point1, point2 in zip(points, points2):
    length = 600
    point1_rpt = point1.repeat(length, 1).permute(1, 0)
    point2_rpt = point2.repeat(length, 1).permute(1, 0)
    ramp = torch.linspace(0, 1, steps=length)
    ramp_rev = torch.linspace(1, 0, steps=length)
    rands = ramp * torch.rand_like(point1_rpt)
    rands_rev = ramp_rev * torch.rand_like(point2_rpt)
    point_rpt = (1 - ramp) * (point1_rpt + rands) + ramp * (point2_rpt + rands_rev)
    latents.append(point_rpt)

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

print("rendering audio...")
z_stack = projector.encode(stack)

z_sum = z_stack.sum(0)

latent_sum = projector.decode(z_sum).permute(1, 0)

audio = encdec.decode(latent_sum)

# encode into projections space

torchaudio.save(f"cloud.wav", audio, 44100)
