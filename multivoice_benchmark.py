import argparse

import librosa
import torch
import torchaudio

from torch.utils.data import DataLoader, ChainDataset

import random

from tqdm import tqdm

from music2latent import EncoderDecoder

from oplas.data import MTGJamendoStreamSingle
from oplas.losses import get_loss_fn
from oplas.mixing import mix_and_encode, mix_single
from oplas.models import Music2Latent, Projector

from helper import get_projector

from statistics import fmean

import matplotlib.pyplot as plt

@torch.no_grad()
def test(projector, device, dl, encoder, step=None):
    """Validation function, slightly simplified for clarity."""
    projector.eval()  # Set model to evaluation mode

    losses = []

    NUM_SAMPLES = 100
    dl_iter = iter(dl)

    # tbatch = tqdm(dl, desc="valid", smoothing=0, leave=False)
    for _ in tqdm(range(NUM_SAMPLES)):
        stems = next(dl_iter)
        # Mix and encode
        stems = stems.to(device)
        mixes = mix_single(stems.unsqueeze(0), encoder, static_mix=False)

        # Call the new centralized loss function
        recon_loss = calculate_losses(projector, mixes["ys"], mixes["y_mix"], step)
        losses.append(recon_loss.detach())

    loss_avg = fmean(losses)
    return loss_avg


def calculate_losses(projector, ys, y_mix, step=0):
    """
    Calculate Reconstruction Loss - This is the main thing we want to
    compare, the vicreg loss stuff was for regularization

    """

    # full audio mix
    z_mix, y_hat_mix, *params_mix = projector(y_mix.permute(0, 2, 1))
    z_mix = z_mix.permute(0, 2, 1)
    y_hat_mix = y_hat_mix.permute(0, 2, 1)

    # stems
    ys_tensor = ys
    ys_permuted = ys_tensor.permute(0, 1, 3, 2)
    z_stems, y_hats, *params_stems = projector(ys_permuted)

    # z-space projection mix
    z_sum_decoded = torch.sum(z_stems, dim=1)
    y_sum = projector.decode(z_sum_decoded).permute(0, 2, 1)
    z_sum = z_sum_decoded.permute(0, 2, 1).contiguous()
    y_hats = y_hats.permute(0, 1, 3, 2)

    # losses
    loss_fn = get_loss_fn("pseudo-huber")
    y_loss = loss_fn(ys_tensor, y_hats)
    y_mix_loss = loss_fn(y_mix, y_hat_mix)
    sum_loss = loss_fn(y_mix, y_sum)

    inv_loss = loss_fn(z_mix, z_sum)

    recon_loss = y_mix_loss + y_loss + inv_loss + sum_loss
    return recon_loss

def main():
    parser = argparse.ArgumentParser(description="Benchmark different models")
    parser.add_argument(
        "-n",
        "--nvoices",
        type=int,
        default=16,
        help="up to how many voices to combine",
    )

    parser.add_argument(
        "-k",
        "--keys",
        nargs="+",
        required=True,
        type=str,
        help="artifact model artifact keys ",
    )

    parser.add_argument(
        "-v",
        "--versions",
        nargs="+",
        type=str,
        help="artifact model versions (i.e. v121 or latest) ",
    )

    args = parser.parse_args()

    if args.versions is not None and  len(args.keys) != len(args.versions):
        print("KEYS MUST BE SAME LENGTH AS VERSIONS")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MTGJamendoStreamSingle(
        data_dir="/scratch/users/nshaheed/mtg-jamendo-wav",
        load_frac=0.2,
    )

    losses = {}

    encoder = Music2Latent().to(device)
    encoder.eval()  # Encoder is always frozen

    # run for n iterations (regarless of number of voices) 
    # just load enough for doing nvoices * niters (and cutoff before that)
    for (key, version) in tqdm(zip(args.keys, args.versions)):
        for n_voices in tqdm(range(args.nvoices)):
            projector = get_projector(artifact_key=key, version=version, device=device)
            projector = projector.to(device)

            
            test_dl = DataLoader(
                dataset,
                batch_size=n_voices+1,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
                shuffle=True,
            )

            recon_loss = test(projector, device, test_dl, encoder)
            
            dict_key = f"{key}_{version}"
            
            if dict_key in losses:
                losses[dict_key].append(recon_loss)
            else:
                losses[dict_key] = [recon_loss]


    # Create a new figure
    plt.figure(figsize=(8, 6))

    # Plot each list in the dictionary
    for key, values in losses.items():
        plt.plot(range(len(values)), values, marker='o', label=key)

    # Add labels and title
    plt.xlabel("Num Voices")
    plt.ylabel("Reconstruction Loss")
    plt.title("Recon loss for number of voices")
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig("multivoice_benchmark.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
