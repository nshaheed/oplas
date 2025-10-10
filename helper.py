from oplas.models import Projector
from latent_effects import Gain

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
from torch.linalg import norm

import wandb


def get_projector(artifact_key="uoznl0zb", version="latest", device=None):
    api = wandb.Api()

    artifact = api.artifact(
        f"nshaheed-stanford-university/oplas/model-ckpt-{artifact_key}:{version}"
    )
    artifact_dir = artifact.download()
    checkpoint_path = Path(artifact_dir) / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # load project with run info
    config = artifact.logged_by().config
    projector = Projector(
        in_dims=64,
        out_dims=config["projector_dims"],
        num_inner_layers=config["num_inner_layers"],
        hidden_dims_scale=config["hidden_dims_scale"],
    )
    projector.load_state_dict(checkpoint["model_state_dict"])

    return projector


def get_gain(artifact_key="lgkjzlyq", device=None):
    api = wandb.Api()

    artifact = api.artifact(
        f"nshaheed-stanford-university/my-first-sweep/model-ckpt-{artifact_key}:latest"
    )
    artifact_dir = artifact.download()
    checkpoint_path = Path(artifact_dir) / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # load project with run info
    config = artifact.logged_by().config
    gain = Gain(
        in_dims=65,
        out_dims=64,
        hidden_dims_scale=config["hidden_dims_scale"],
        num_inner_layers=config["num_inner_layers"],
        resid=False,
    )

    gain.load_state_dict(checkpoint["model_state_dict"])

    return gain


def plot_audio(wav):
    # Assuming wav is your audio array (1D numpy array)
    # If you know the sample rate:
    sr = 44100  # replace with your actual sample rate

    # Create a time axis in seconds
    time = np.linspace(0, len(wav) / sr, num=len(wav))

    # Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, wav)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.show()


def plot_audio2(wav1, wav2, sr=44100):
    # Create time axes
    time1 = np.linspace(0, len(wav1) / sr, num=len(wav1))
    time2 = np.linspace(0, len(wav2) / sr, num=len(wav2))

    # Set up subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot first waveform
    axs[0].plot(time1, wav1)
    axs[0].set_title("Waveform 1")
    axs[0].set_ylabel("Amplitude")

    # Plot second waveform
    axs[1].plot(time2, wav2, color="orange")
    axs[1].set_title("Waveform 2")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


# adapted to PyTorch from:
# https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
# most of the extra complexity is to support:
# - many-dimensional vectors
# - v0 or v1 with last dim all zeroes, or v0 ~colinear with v1
#   - falls back to lerp()
#   - conditional logic implemented with parallelism rather than Python loops
# - many-dimensional tensor for t
#   - you can ask for batches of slerp outputs by making t more-dimensional than the vectors
#   -   slerp(
#         v0:   torch.Size([2,3]),
#         v1:   torch.Size([2,3]),
#         t:  torch.Size([4,1,1]),
#       )
#   - this makes it interface-compatible with lerp()
def slerp(
    v0: FloatTensor, v1: FloatTensor, t: float | FloatTensor, DOT_THRESHOLD=0.9995
):
    """
    Spherical linear interpolation
    Args:
      v0: Starting vector
      v1: Final vector
      t: Float value between 0.0 and 1.0
      DOT_THRESHOLD: Threshold for considering the two vectors as
                              colinear. Not recommended to alter this.
    Returns:
        Interpolation vector between v0 and v1
    """
    assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

    # Normalize the vectors to get the directions and angles
    v0_norm: FloatTensor = norm(v0, dim=-1)
    v1_norm: FloatTensor = norm(v1, dim=-1)

    v0_normed: FloatTensor = v0 / v0_norm.unsqueeze(-1)
    v1_normed: FloatTensor = v1 / v1_norm.unsqueeze(-1)

    # Dot product with the normalized vectors
    dot: FloatTensor = (v0_normed * v1_normed).sum(-1)
    dot_mag: FloatTensor = dot.abs()

    # if dp is NaN, it's because the v0 or v1 row was filled with 0s
    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    gotta_lerp: LongTensor = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
    can_slerp: LongTensor = ~gotta_lerp

    t_batch_dim_count: int = max(0, t.dim() - v0.dim()) if isinstance(t, Tensor) else 0
    t_batch_dims: Size = (
        t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size([])
    )
    out: FloatTensor = zeros_like(v0.expand(*t_batch_dims, *[-1] * v0.dim()))

    # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
    if gotta_lerp.any():
        lerped: FloatTensor = lerp(v0, v1, t)

        out: FloatTensor = lerped.where(gotta_lerp.unsqueeze(-1), out)

    # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
    if can_slerp.any():
        # Calculate initial angle between v0 and v1
        theta_0: FloatTensor = dot.arccos().unsqueeze(-1)
        sin_theta_0: FloatTensor = theta_0.sin()
        # Angle at timestep t
        theta_t: FloatTensor = theta_0 * t
        sin_theta_t: FloatTensor = theta_t.sin()
        # Finish the slerp algorithm
        s0: FloatTensor = (theta_0 - theta_t).sin() / sin_theta_0
        s1: FloatTensor = sin_theta_t / sin_theta_0
        slerped: FloatTensor = s0 * v0 + s1 * v1

        out: FloatTensor = slerped.where(can_slerp.unsqueeze(-1), out)

    return out
