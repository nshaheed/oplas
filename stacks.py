from pathlib import Path
from tqdm import tqdm

import librosa
import wandb
from oplas.models import Music2Latent, Projector, ProjectorVAE, VAE
from helper import get_projector, get_gain, plot_audio, plot_audio2, slerp

from music2latent import EncoderDecoder

import torch
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# id = "uoznl0zb"  # fallen-sweep-4

# projector = Projector(in_dims=64, out_dims=256, num_inner_layers=1, hidden_dims_scale=8)

# # run = wandb.init(project="oplas", job_type="inference")
# api = wandb.Api()
# try:
#     artifact = api.artifact(
#         f"nshaheed-stanford-university/oplas/model-ckpt-{id}:latest"
#     )
#     artifact_dir = artifact.download()
#     checkpoint_path = Path(artifact_dir) / "model.pt"
#     checkpoint = torch.load(checkpoint_path, map_location=device)

#     projector.load_state_dict(checkpoint["model_state_dict"])
# except Exception as e:
#     print(f"Could not resume. Starting from scratch. Error: {e}")

projector = get_projector(artifact_key="oauvlx7z", version="v142", device=device) # old 4 voice
# projector = get_projector(artifact_key="00245wks", version="v73", device=device) # new 16 voice


encdec = EncoderDecoder()

# our stems
file_list = [
    "./stems/harmony2_pad/Alto pt. 1_1.wav",
    "./stems/harmony2_pad/Alto pt. 2 (double)_1.wav",
    "./stems/harmony2_pad/Alto pt. 2 [-12]_1.wav",
    "./stems/harmony2_pad/Alto pt. 2_1.wav",
    "./stems/harmony2_pad/Audio 9_bip_1.wav",
    "./stems/harmony2_pad/First Soprano pt. 1_1.wav",
    "./stems/harmony2_pad/First Soprano pt. 2_1.wav",
    "./stems/harmony2_pad/Lowest (bass )_1.wav",
    "./stems/harmony2_pad/Lowest (bass -12 )_1.wav",
    "./stems/harmony2_pad/Lowest (bass -5 cool counter part idea)_1.wav",
    "./stems/harmony2_pad/Second Soprano pt. 2_1.wav",
    "./stems/harmony2_pad/Second Soprano pt. 3_1.wav",
    "./stems/harmony2_pad/Second soprano pt. 1 (double)_1.wav",
    "./stems/harmony2_pad/Second soprano pt. 4_1.wav",
    "./stems/harmony2_pad/Second soprano pt.1_1.wav",
    "./stems/harmony2_pad/low ish (Tenor pt. 2)_1.wav",
    "./stems/harmony2_pad/lower (Tenor pt. 1)_1.wav",
]

file_list = file_list[:10]

latents = []
for file in tqdm(file_list, "encoding files"):
    wv, _ = librosa.load(file, sr=44100)
    latent = encdec.encode(wv).squeeze()
    # breakpoint()
    latents.append(latent)

stack = torch.stack(latents).permute(0, 2, 1)

z_stack = projector.encode(stack)

z_sum = z_stack.sum(0)

latent_sum = projector.decode(z_sum).permute(1, 0)

audio = encdec.decode(latent_sum)

# encode into projections space

torchaudio.save(f"harmony2-sum.wav", audio, 44100)
