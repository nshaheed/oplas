import argparse
from itertools import repeat
from pathlib import Path
import os
import math
import time

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, ChainDataset
from tqdm import tqdm
import torch.multiprocessing as mp

from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder
from ffcv.transforms import ToTensor, ToDevice

import wandb

# Assuming your custom modules are in the python path
from oplas.data import (
    StemChunk,
    StemChunkStream,
    MTGJamendoStreamSingle,
    RandomMixDataset,
    MTGJamendoCache,
)
from oplas.losses import vicreg_loss_fn, get_loss_fn, kld
from oplas.mixing import mix_and_encode, mix_single
from oplas.models import Music2Latent, Projector, ProjectorVAE, VAE
from oplas.helper import save_model, get_scheduler


def train(run, config, checkpoint=None, ignore_max_steps=False, test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    out_dims = config['projector_dims']

    # --- Model, Optimizer, Scheduler ---
    projector = None

    projector = Projector(
        in_dims=64 * 2,  # two subsequent latent dims
        out_dims=out_dims,  # what is this gonna beeee
        num_inner_layers=config['num_inner_layers'],
        hidden_dims_scale=config['hidden_dims_scale'],
    ).to(device)
    encoder = Music2Latent().to(device)
    encoder.eval()  # Encoder is always frozen

    opt = torch.optim.Adam(
        projector.parameters(),
        lr=config['learning_rate'],
        betas=(config['adams_beta1'], 0.999),
        eps=config['adams_epsilon'],
    )

    BATCH_SIZE = config['batch_size']
    NUM_WORKERS = 1
    ORDERING = OrderOption.QUASI_RANDOM

    PIPELINES = {
        "audio": [NDArrayDecoder(), ToTensor(), ToDevice(device)],
    }

    train_dl = Loader(
        "/scratch/users/nshaheed/mtg-jamendo-ffcv-small.beton",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        order=ORDERING,
        pipelines=PIPELINES,
        os_cache=False,  # can't cache bc it's too big to fit in memory
        drop_last=True,
    )
    # val_dl = Loader(
    #     "/scratch/users/nshaheed/mtg-jamendo-ffcv-val.beton",
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     order=ORDERING,
    #     pipelines=PIPELINES,
    #     os_cache=False,  # can't cache bc it's too big to fit in memory
    #     drop_last=True,
    # )

    # --- Training Loop ---
    projector.train()

    step = 0

    max_epochs = 10
    epochs = tqdm(range(max_epochs), desc="epoch", smoothing=0)

    # The actual training loop
    for i in epochs:
        tbatch = tqdm(train_dl, desc="steps", smoothing=0, leave=False)
        done = False
        for (batch,) in tbatch:
            if done:
                return
            done = True

            opt.zero_grad()

            # get two adjacent values in a batch ( i don't think this is really necessary?
            b, t = batch.shape

            # (Pdb) audio.shape
            # torch.Size([32, 259584])
            # (Pdb) batch.shape
            # torch.Size([32, 262144])

            # latents = encoder.encode(batch)

            audio = batch[0]
            # encode the full audio
            latents = encoder.encode(audio)

            # decode as single latent
            full_audio_reconstructed = encoder.decode(latents)

            individual_latents_reconstructed = []
            # break into individual latents and save as separate files
            div = 1
            for i in range(latents.shape[-1]//div):
                print(f'{div*i=}, {div*i+div=}')
                # latent_audio = encoder.decode(latents[:,:,8*i:8*i+8])
                latent_audio = encoder.decode(latents[:,:,div*i:div*i+div])
                individual_latents_reconstructed.append(latent_audio)

            breakpoint()

            torchaudio.save('./test_audio/full_audio.wav', full_audio_reconstructed, 44100)

            for i, aud in enumerate(individual_latents_reconstructed):
                torchaudio.save(f'./test_audio/latent_audio_{i:03}.wav', aud, 44100)
            
            # breakpoint()
            # for i in range(1,100):
            #     latents = torch.rand(32,64,i)
            #     audio = encoder.decode(latents)
            #     print(f'Latents {i}: {audio.shape}, sample/latent: {audio.shape[1]/i}')
            # breakpoint()


def main():

    parser = argparse.ArgumentParser(description="Training script with wandb")

    # Three modes: sweep, single, resume
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="W&B sweep ID to run as an agent (sweep mode).",
    )
    parser.add_argument(
        "--resume_run_id",
        type=str,
        default=None,
        help="W&B run ID to resume training (resume mode).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint when resuming.",
    )

    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--num_inner_layers", type=int, default=6)
    parser.add_argument("--hidden_dims_scale", type=int, default=6)
    parser.add_argument(
        "--projector_dims", type=int, default=128, help="num of dims in projection space"
    )

    parser.add_argument("--var_coeff", type=float, default=1.0)
    parser.add_argument("--inv_coeff", type=float, default=1.0)
    parser.add_argument("--cov_coeff", type=float, default=1.0)

    parser.add_argument("--adams_beta1", type=float, default=0.9)
    parser.add_argument("--adams_epsilon", type=float, default=1e-8)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_stems", type=int, default=8)
    parser.add_argument("--num_chunks", type=int, default=10)
    parser.add_argument(
        "--data_dir", type=str, default="/scratch/users/nshaheed/musdb18"
    )

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--load_frac", type=float, default=1.0)
    parser.add_argument("--checkpoint_every", type=int, default=400)
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        help="which loss function to use [mse,pseudo-huber]",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        help="which loss scheudler to use [None,cosine]",
    )
    parser.add_argument(
        "--ignore_max_steps",
        action=argparse.BooleanOptionalAction,
        help="ignore max step and train forever",
    )
    parser.add_argument("--arch", type=str, help="use 'vae' or autoencoder(default)")
    parser.add_argument("--kld_warmup", type=int, help="warmup of kld in vae")
    parser.add_argument("--augment", action="store_true")
    parser.set_defaults(augment=False)
    parser.add_argument("--test", action="store_true")
    parser.set_defaults(augment=False)
    args = parser.parse_args()

    config = {
        "learning_rate": args.learning_rate,
        "adams_beta1": args.adams_beta1,
        "adams_epsilon": args.adams_epsilon,
        "num_inner_layers": args.num_inner_layers,
        "hidden_dims_scale": args.hidden_dims_scale,
        "projector_dims": args.projector_dims,
        "var_coeff": args.var_coeff,
        "inv_coeff": args.inv_coeff,
        "cov_coeff": args.cov_coeff,
        "batch_size": args.batch_size,
        "data_dir": args.data_dir,
        "max_steps": args.max_steps,
        "load_frac": args.load_frac,
        "checkpoint_every": args.checkpoint_every,
        "val_every": args.val_every,
        "loss": args.loss,
        "scheduler": args.scheduler,
        "augment": args.augment,
        "arch": args.arch,
        "kld_warmup": args.kld_warmup,
        "num_stems": args.num_stems,
        "num_chunks": args.num_chunks,
    }
    
    train(None, config)

if __name__ == "__main__":
    main()
