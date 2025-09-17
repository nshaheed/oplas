# train_sweep.py

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
from oplas.data import StemChunk, StemChunkStream, MTGJamendoStream, RandomMixDataset
from oplas.losses import vicreg_loss_fn, get_loss_fn, kld
from oplas.mixing import mix_and_encode
from oplas.models import Music2Latent, Projector, ProjectorVAE, VAE
from oplas.helper import save_model, get_scheduler



@torch.no_grad()
def validate(projector, device, val_dl, encoder, config, step=None):
    """Validation function, slightly simplified for clarity."""
    projector.eval()  # Set model to evaluation mode
    vbatch_iter = iter(val_dl)
    try:
        batch = next(vbatch_iter).to(device)
    except StopIteration:
        print("Validation dataloader is empty.")
        return {}

    # --- Your validation logic remains the same ---
    mixes = mix_and_encode(batch, encoder, debug=False)
    y_mix = mixes["y_mix"].float()
    z_sum = None

    z_mix, y_hat_mix, *params = projector(y_mix.permute(0, 2, 1))
    z_mix = z_mix.permute(0, 2, 1)
    y_hat_mix = y_hat_mix.permute(0, 2, 1)

    ys_tensor = torch.stack(mixes["ys"], dim=1).to(torch.float32)
    B, S, D, T = ys_tensor.shape
    ys_permuted = ys_tensor.permute(0, 1, 3, 2)
    z_stems, y_hats, *params = projector(ys_permuted)

    z_sum = torch.sum(z_stems, dim=1)  # Sum over stems dimension
    y_sum = projector.decode(z_sum).permute(0, 2, 1)
    z_sum = z_sum.permute(0, 2, 1).contiguous()
    y_hats = y_hats.permute(0, 1, 3, 2)

    loss_fn = get_loss_fn(config.loss)
    vicreg_loss = vicreg_loss_fn(z_sum, z_mix)
    y_loss = loss_fn(ys_tensor, y_hats)
    y_mix_loss = loss_fn(y_mix, y_hat_mix)
    sum_loss = loss_fn(y_mix, y_sum)

    kld_loss = 0
    if config.arch == "vae":
        mu, logvar = params
        kld_loss = kld(mu, logvar, step, config.kld_warmup)

    var_loss = config.var_coeff * vicreg_loss["var_loss"]
    # inv_loss = config.inv_coeff * vicreg_loss["inv_loss"]
    inv_loss = loss_fn(z_mix, z_sum)
    cov_loss = config.cov_coeff * vicreg_loss["cov_loss"]

    loss = var_loss + inv_loss + cov_loss + y_loss + y_mix_loss + sum_loss + kld_loss

    # testing z_sum back to y_sum

    log = {
        "val/loss": loss.detach(),
        "val/var_loss": var_loss.detach(),
        "val/inv_loss": inv_loss.detach(),
        "val/cov_loss": cov_loss.detach(),
        "val/y_loss": y_loss.detach(),
        "val/y_mix_loss": y_mix_loss.detach(),
        "val/sum_loss": sum_loss.detach(),
        "val/recon_loss": y_mix_loss.item()
        + y_loss.item()
        + inv_loss.item()
        + sum_loss.item(),
    }

    if config.arch == "vae":
        log = log | {"val/kld_loss": kld_loss.item()}

    # actually generate some audio examples
    latent = projector.decode(z_sum.permute(0, 2, 1))  # swap last two dims
    latent = latent.permute(0, 2, 1)  # have to swap it back

    audio = encoder.decode(latent).cpu()

    log = log | {
        "val/0/z_mix": wandb.Audio(
            audio[0],
            caption="decoding of audio mixed in the z domain",
            sample_rate=44100,
        ),
        "val/0/orig": wandb.Audio(
            mixes["g_mix"][0, :, 0].cpu(),
            # batch[0, 0, :, 0].cpu(),
            caption="original mix",
            sample_rate=44100,
        ),
        "val/1/z_mix": wandb.Audio(
            audio[1],
            caption="decoding of audio mixed in the z domain",
            sample_rate=44100,
        ),
        "val/1/orig": wandb.Audio(
            mixes["g_mix"][1, :, 0].cpu(),
            # batch[1, 0, :, 0].cpu(),
            caption="original mix",
            sample_rate=44100,
        ),
        "val/2/z_mix": wandb.Audio(
            audio[2],
            caption="decoding of audio mixed in the z domain",
            sample_rate=44100,
        ),
        "val/2/orig": wandb.Audio(
            mixes["g_mix"][2, :, 0].cpu(),
            # batch[2, 0, :, 0].cpu(),
            caption="original mix",
            sample_rate=44100,
        ),
    }

    projector.train()  # Set model back to training mode
    return log


def train(run, config, checkpoint=None, ignore_max_steps=False, test=False):
    """Main training function wrapped for W&B sweeps."""
    # Initialize a new wandb run
    # run = wandb.init()
    # config = wandb.config  # Get hyperparams from the sweep

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # config.load_frac = 0.02
    # config.batch_size = 1
    # config.num_inner_layers = 1
    # config.hidden_dims_scale = 1

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
    # TODO use fixed learning rate for now
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     opt, max_lr=config.learning_rate, total_steps=config.max_steps
    # )
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

    # --- Data Loading ---
    # train_dataset = StemChunk(data_dir=config.data_dir, load_frac=config.load_frac)
    stems_dataset = StemChunkStream(
        data_dir=config.data_dir, load_frac=config.load_frac, augment=config.augment
    )
    mtg_dataset = MTGJamendoStream(augment=config.augment)
    train_dataset = RandomMixDataset([stems_dataset, mtg_dataset])
    val_dataset = StemChunkStream(
        data_dir=config.data_dir, subset="test", load_frac=config.load_frac
    )

    if test:
        # for debugging on sh_dev
        train_dl = DataLoader(  # do 12 workers?
            stems_dataset, batch_size=config.batch_size, num_workers=0, pin_memory=True
        )
        val_dl = DataLoader(
            val_dataset, batch_size=config.batch_size, num_workers=0, pin_memory=True
        )
    else:
        train_dl = DataLoader(  # do 12 workers?
            train_dataset, batch_size=config.batch_size, num_workers=16, pin_memory=True
        )
        val_dl = DataLoader(
            val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True
        )

    # --- Training Loop ---
    projector.train()
    tbatch = tqdm(
        iter(train_dl),
        initial=start_step,
        total=config.max_steps,
        unit="batch",
        desc="training",
    )

    # repeat one batch infinitely
    if test:
        tbatch = tqdm(
            repeat(next(iter(train_dl))),
            initial=start_step,
            total=config.max_steps,
            unit="batch",
            desc="training",
        )
        # tbatch = repeat(next(iter(tbatch)))

    mixes = None

    dataload_time = time.time()
    for batch in tbatch:
        print(f'Data loading time: {time.time() - dataload_time}s')
        step = tbatch.n
        if step >= config.max_steps and not ignore_max_steps:
            break
        batch = batch.to(device)
        # batch = torch.rand_like(batch)
        opt.zero_grad()

        # augment with effects
        # if config.augment:
        #     batch = augment_effects(batch, sample_rate=44100)

        encoding_time = time.time()
        with torch.no_grad():
            if test and mixes is None:
                mixes = mix_and_encode(batch, encoder, static_mix=test, debug=False)
            if not test:
                mixes = mix_and_encode(batch, encoder, static_mix=test, debug=False)
            y_mix = mixes["y_mix"].to(torch.float32)

        print(f'Encoding time: {time.time() - encoding_time}s')

        # Vectorized projection logic (more efficient)
        z_mix, y_hat_mix, *params = projector(y_mix.permute(0, 2, 1))
        z_mix = z_mix.permute(0, 2, 1)
        y_hat_mix = y_hat_mix.permute(0, 2, 1)

        # TODO validate
        ys_tensor = torch.stack(mixes["ys"], dim=1).to(torch.float32)
        B, S, D, T = ys_tensor.shape
        ys_permuted = ys_tensor.permute(0, 1, 3, 2)
        z_stems, y_hats, *params = projector(ys_permuted)
        # z_stems = z_stems_flat.reshape(B, S, T, D).permute(0, 1, 3, 2)
        # y_hats_tensor = y_hats_flat.reshape(B, S, T, D).permute(0, 1, 3, 2)
        z_sum = torch.sum(z_stems, dim=1)
        y_sum = projector.decode(z_sum).permute(0, 2, 1)
        z_sum = z_sum.permute(0, 2, 1).contiguous()
        y_hats = y_hats.permute(0, 1, 3, 2)

        # --- Loss Calculation ---
        vicreg_loss = vicreg_loss_fn(z_sum, z_mix)
        y_loss = loss_fn(ys_tensor, y_hats)
        y_mix_loss = loss_fn(y_mix, y_hat_mix)
        sum_loss = loss_fn(y_mix, y_sum)

        kld_loss = torch.tensor(0)
        if config.arch == "vae":
            mu, logvar = params
            kld_loss = kld(mu, logvar, step, config.kld_warmup)

        var_loss = config.var_coeff * vicreg_loss["var_loss"]
        # inv_loss = config.inv_coeff * vicreg_loss["inv_loss"]
        inv_loss = loss_fn(z_mix, z_sum)
        cov_loss = config.cov_coeff * vicreg_loss["cov_loss"]

        loss = (
            var_loss + inv_loss + cov_loss + y_loss + y_mix_loss + sum_loss + kld_loss
        )

        loss = y_mix_loss

        loss.backward()
        opt.step()
        if scheduler:
            scheduler.step()

        # --- Logging ---
        tbatch.set_postfix(loss=loss.item(), sum_loss=sum_loss.item())
        log_dict = {
            "train/loss": loss.item(),
            "train/var_loss": var_loss.item(),
            "train/inv_loss": inv_loss.item(),
            "train/cov_loss": cov_loss.item(),
            "train/y_loss": y_loss.item(),
            "train/y_mix_loss": y_mix_loss.item(),
            "train/sum_loss": sum_loss.detach(),
            "train/recon_loss": y_mix_loss.item()
            + y_loss.item()
            + inv_loss.item()
            + sum_loss.item(),
        }

        if config.arch == "vae":
            log_dict = log_dict | {"train/kld_loss": kld_loss.item()}

        if scheduler:
            log_dict = log_dict | {"train/learning_rate": scheduler.get_last_lr()[0]}

        if step % config.val_every == 0:
            latent = projector.decode(z_sum.permute(0, 2, 1))  # swap last two dims
            latent = latent.permute(0, 2, 1)  # have to swap it back
            audio = encoder.decode(latent).cpu()

            log_dict = log_dict | {
                "train/0/z_mix": wandb.Audio(
                    audio[0],
                    caption="decoding of audio mixed in the z domain",
                    sample_rate=44100,
                ),
                "train/0/orig": wandb.Audio(
                    mixes["g_mix"][0, :, 0].cpu(),
                    # batch[0, 0, :, 0].cpu(),
                    caption="original mix",
                    sample_rate=44100,
                ),
                "train/1/z_mix": wandb.Audio(
                    audio[1],
                    caption="decoding of audio mixed in the z domain",
                    sample_rate=44100,
                ),
                "train/1/orig": wandb.Audio(
                    mixes["g_mix"][1, :, 0].cpu(),
                    # batch[1, 0, :, 0].cpu(),
                    caption="original mix",
                    sample_rate=44100,
                ),
                "train/2/z_mix": wandb.Audio(
                    audio[2],
                    caption="decoding of audio mixed in the z domain",
                    sample_rate=44100,
                ),
                "train/2/orig": wandb.Audio(
                    mixes["g_mix"][2, :, 0].cpu(),
                    # batch[2, 0, :, 0].cpu(),
                    caption="original mix",
                    sample_rate=44100,
                ),
            }

            val_log = validate(projector, device, val_dl, encoder, config, step=step)
            log_dict.update(val_log)

        run.log(log_dict)

        if step % config.checkpoint_every == 0 and step > 0:
            checkpoint_path = save_model(projector, step, opt, scheduler, loss, run.id)
            # Log checkpoint as a versioned artifact
            artifact = wandb.Artifact(f"model-ckpt-{run.id}", type="model")
            artifact.add_file(local_path=checkpoint_path, name="model.pt")
            run.log_artifact(artifact)
        
        dataload_time = time.time()


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
