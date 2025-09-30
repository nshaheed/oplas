import argparse
from itertools import repeat
from pathlib import Path
import os
import math
import time

import numpy as np
import torch
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

    out_dims = config.projector_dims

    # --- Model, Optimizer, Scheduler ---
    projector = None

    projector = Projector(
        in_dims=64 * 2,  # two subsequent latent dims
        out_dims=out_dims,  # what is this gonna beeee
        num_inner_layers=config.num_inner_layers,
        hidden_dims_scale=config.hidden_dims_scale,
    ).to(device)
    encoder = Music2Latent().to(device)
    encoder.eval()  # Encoder is always frozen

    opt = torch.optim.Adam(
        projector.parameters(),
        lr=config.learning_rate,
        betas=(config.adams_beta1, 0.999),
        eps=config.adams_epsilon,
    )

    BATCH_SIZE = config.batch_size
    NUM_WORKERS = 4
    ORDERING = OrderOption.QUASI_RANDOM

    PIPELINES = {
        "audio": [NDArrayDecoder(), ToTensor(), ToDevice(device)],
    }

    train_dl = Loader(
        "./data/mtg-jamendo-ffcv-small.beton",
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

    step = start_step

    max_epochs = 10
    epochs = tqdm(range(max_epochs), desc="epoch", smoothing=0)

    # The actual training loop
    for i in epochs:
        tbatch = tqdm(train_dl, desc="steps", smoothing=0, leave=False)

        for (batch,) in tbatch:
            opt.zero_grad()

            # get two adjacent values in a batch ( i don't think this is really necessary?
            b, t = batch.shape
            breakpoint()


def main():
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
        "kld_warmup": args.kld_warmupo,
        "num_stems": args.num_stems,
        "num_chunks": args.num_chunks,
    }

    train(None, config)
