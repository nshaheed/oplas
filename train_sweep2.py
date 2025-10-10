import argparse
from itertools import islice, repeat
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

import wandb

# Assuming your custom modules are in the python path
from oplas.data import MTGJamendo
from oplas.losses import vicreg_loss_fn, get_loss_fn, kld
from oplas.mixing import mix_latents
from oplas.models import Music2Latent, Projector, ProjectorVAE, VAE
from oplas.helper import save_model, get_scheduler


def train(run, config, checkpoint=None, ignore_max_steps=False, test=False):
    """Main training function wrapped for W&B sweeps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # handle old models where this wasn't a variable
    out_dims = config.projector_dims if "projector_dims" in config else 64

    # --- Model, Optimizer, Scheduler ---
    projector = None

    if config.arch == "vae":
        # projector = ProjectorVAE(
        #     in_dims=64,
        #     out_dims=out_dims,
        #     num_inner_layers=config.num_inner_layers,
        #     hidden_dims_scale=config.hidden_dims_scale,
        # ).to(device)

        projector = VAE(in_dims=64, out_dims=out_dims).to(device)
    else:
        projector = Projector(
            in_dims=64,
            out_dims=out_dims,
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
    scheduler = get_scheduler(opt, config)
    loss_fn = get_loss_fn(config.loss)

    start_step = 0

    # --- W&B Resuming Logic ---
    if wandb.run.resumed:
        print("Resuming run from a checkpoint...")
        # wandb.restore() downloads the file and returns a file path
        # It's safer to restore the most recent checkpoint. We'll log it as an artifact.
        try:
            artifact = run.use_artifact(f"model-ckpt-{run.id}:latest")
            artifact_dir = artifact.download()
            checkpoint_path = Path(artifact_dir) / "model.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device)

            projector.load_state_dict(checkpoint["model_state_dict"])
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint["scheduler"]:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_step = checkpoint["step"] + 1
            print(f"Resumed successfully from step {start_step}.")
        except Exception as e:
            print(f"Could not resume. Starting from scratch. Error: {e}")

    # Data loading
    mtg_dataset = MTGJamendo()

    train_dataset, val_dataset = torch.utils.data.random_split(mtg_dataset, [0.9, 0.1])

    train_dl = DataLoader(
        train_dataset, batch_size=config.batch_size, num_workers=0, pin_memory=True
    )
    val_dl = DataLoader(
        val_dataset, batch_size=config.batch_size, num_workers=0, pin_memory=True
    )

    step = 0
    while True:
        if step >= config.max_steps and not ignore_max_steps:
            break
        for batch in tqdm(train_dl):
            if step >= config.max_steps and not ignore_max_steps:
                break

            batch = batch.to(device)
            opt.zero_grad()

            mix_latents(batch, encoder)

            step += 1


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
        "--projector_dims", type=int, default=64, help="num of dims in projection space"
    )

    parser.add_argument("--var_coeff", type=float, default=1.0)
    parser.add_argument("--inv_coeff", type=float, default=1.0)
    parser.add_argument("--cov_coeff", type=float, default=1.0)

    parser.add_argument("--adams_beta1", type=float, default=0.9)
    parser.add_argument("--adams_epsilon", type=float, default=1e-8)

    parser.add_argument("--batch_size", type=int, default=32)
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

    # --- Mode 1: Sweep run ---
    if args.sweep_id:

        def sweep_train():
            with wandb.init() as run:
                config = wandb.config  # get hyperparams from sweep
                train(
                    run, config, ignore_max_steps=args.ignore_max_steps, test=args.test
                )

        wandb.agent(args.sweep_id, function=sweep_train)

    # --- Mode 2: Resume run ---
    elif args.resume_run_id:
        project = "oplas"
        with wandb.init(project=project, id=args.resume_run_id, resume="must") as run:
            config = run.config  # restore config from original run
            train(
                run,
                config,
                checkpoint=args.checkpoint,
                ignore_max_steps=args.ignore_max_steps,
                test=args.test,
            )

    # --- Mode 3: Single run ---
    else:
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
        }
        with wandb.init(config=config) as run:
            train(
                run, run.config, ignore_max_steps=args.ignore_max_steps, test=args.test
            )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Run a W&B sweep agent for the oplas model."
    # )
    # parser.add_argument(
    #     "sweep_id",
    #     type=str,
    #     help="wandb sweep id",
    # )
    # args = parser.parse_args()

    # wandb.agent(args.sweep_id, function=train, count=1)
    # train(args)
    mp.set_start_method("spawn", force=True)

    main()
