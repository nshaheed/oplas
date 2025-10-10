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

from torch_lr_finder import LRFinder

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



def calculate_losses(projector, ys, y_mix, config, step=0):
    """
    Pass data through projector and calculate losses
    """

    # full audio mix
    # y_mix = mixes["y_mix"].to(torch.float32)
    z_mix, y_hat_mix, *params_mix = projector(y_mix.permute(0, 2, 1))
    z_mix = z_mix.permute(0, 2, 1)
    y_hat_mix = y_hat_mix.permute(0, 2, 1)

    # stems
    # ys_tensor = torch.stack(mixes["ys"], dim=0).to(torch.float32)
    ys_tensor = ys
    ys_permuted = ys_tensor.permute(0, 1, 3, 2)
    z_stems, y_hats, *params_stems = projector(ys_permuted)

    # z-space projection mix
    z_sum_decoded = torch.sum(z_stems, dim=1)
    y_sum = projector.decode(z_sum_decoded).permute(0, 2, 1)
    z_sum = z_sum_decoded.permute(0, 2, 1).contiguous()
    y_hats = y_hats.permute(0, 1, 3, 2)

    # losses
    loss_fn = get_loss_fn(config.loss)
    vicreg_loss = vicreg_loss_fn(z_sum, z_mix)
    y_loss = loss_fn(ys_tensor, y_hats)
    y_mix_loss = loss_fn(y_mix, y_hat_mix)
    sum_loss = loss_fn(y_mix, y_sum)

    var_loss = config.var_coeff * vicreg_loss["var_loss"]
    inv_loss = loss_fn(z_mix, z_sum)
    cov_loss = config.cov_coeff * vicreg_loss["cov_loss"]

    total_loss = (
        var_loss + inv_loss + cov_loss + y_loss + y_mix_loss + sum_loss
    )

    # returns loss dictionary and z_sum
    return {
        "total_loss": total_loss,
        "var_loss": var_loss,
        "inv_loss": inv_loss,
        "cov_loss": cov_loss,
        "y_loss": y_loss,
        "y_mix_loss": y_mix_loss,
        "sum_loss": sum_loss,
        "recon_loss": y_mix_loss + y_loss + inv_loss + sum_loss,
    }, z_sum

def criterion(result, label):
    """
    for use with LRFinder, needs to just work
    """

    # what is needed (output, labels):
    # - z_sum, z_mix
    # - ys_tensor, y_hats
    # - y_mix, y_hat_mix
    # - y_mix, y_sum

    # projector(y_mix):
    # returns 

    # full audio mix
    # y_mix = mixes["y_mix"].to(torch.float32)
    z_mix, y_hat_mix, *params_mix = projector(y_mix.permute(0, 2, 1))
    z_mix = z_mix.permute(0, 2, 1)
    y_hat_mix = y_hat_mix.permute(0, 2, 1)

    # stems
    # ys_tensor = torch.stack(mixes["ys"], dim=0).to(torch.float32)
    ys_tensor = ys
    ys_permuted = ys_tensor.permute(0, 1, 3, 2)
    z_stems, y_hats, *params_stems = projector(ys_permuted)

    # z-space projection mix
    z_sum_decoded = torch.sum(z_stems, dim=1)
    y_sum = projector.decode(z_sum_decoded).permute(0, 2, 1)
    z_sum = z_sum_decoded.permute(0, 2, 1).contiguous()
    y_hats = y_hats.permute(0, 1, 3, 2)

    # losses
    loss_fn = get_loss_fn(config.loss)
    vicreg_loss = vicreg_loss_fn(z_sum, z_mix)
    y_loss = loss_fn(ys_tensor, y_hats)
    y_mix_loss = loss_fn(y_mix, y_hat_mix)
    sum_loss = loss_fn(y_mix, y_sum)

    var_loss = config.var_coeff * vicreg_loss["var_loss"]
    inv_loss = loss_fn(z_mix, z_sum)
    cov_loss = config.cov_coeff * vicreg_loss["cov_loss"]

    total_loss = (
        var_loss + inv_loss + cov_loss + y_loss + y_mix_loss + sum_loss
    )
    return total_loss # only return loss here

    # returns loss dictionary and z_sum
    return {
        "total_loss": total_loss,
        "var_loss": var_loss,
        "inv_loss": inv_loss,
        "cov_loss": cov_loss,
        "y_loss": y_loss,
        "y_mix_loss": y_mix_loss,
        "sum_loss": sum_loss,
        "recon_loss": y_mix_loss + y_loss + inv_loss + sum_loss,
    }, z_sum

def detach_losses(loss):
    """ detach losses from graph and then move to cpu (for validation) """
    new_loss = {}

    for key, value in loss.items():
        if isinstance(value, torch.Tensor):
            new_loss[key] = value.detach().cpu()
        else:
            new_loss[key] = value

    return new_loss




@torch.no_grad()
def validate(projector, device, val_dl, encoder, config, step=None):
    """Validation function, slightly simplified for clarity."""
    projector.eval()  # Set model to evaluation mode

    losses = []
    z_sum = None
    stems = None
    tbatch = tqdm(val_dl, desc="valid", smoothing=0, leave=False)
    for stems, mix in tbatch:
        # Call the new centralized loss function
        loss_start = time.time()
        loss_dict, z_sum = calculate_losses(projector, stems, mix, config, step)
        loss_end = time.time() - loss_start

        tbatch.set_postfix(loss=loss_end)

        losses.append(detach_losses(loss_dict))

    # take average of losses
    loss_avg = {}

    # sum losses
    for loss_dict in losses:
        for key, value in loss_dict.items():
            if key in loss_avg:
                loss_avg[key] += value
            else:
                loss_avg[key] = value

    # calc mean
    for key, value in loss_avg.items():
        loss_avg[key] = value / len(losses)

    loss = loss_avg["total_loss"]
    log = {
        "val/loss": loss_avg["total_loss"].detach(),
        "val/var_loss": loss_avg["var_loss"].detach(),
        "val/inv_loss": loss_avg["inv_loss"].detach(),
        "val/cov_loss": loss_avg["cov_loss"].detach(),
        "val/y_loss": loss_avg["y_loss"].detach(),
        "val/y_mix_loss": loss_avg["y_mix_loss"].detach(),
        "val/sum_loss": loss_avg["sum_loss"].detach(),
        "val/recon_loss": loss_avg["recon_loss"].item(),
    }

    projector.train()  # Set model back to training mode
    return log




def train(run, config, checkpoint=None, ignore_max_steps=False, test=False):
    """Main training function wrapped for W&B sweeps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # handle old models where this wasn't a variable
    out_dims = config.projector_dims if "projector_dims" in config else 64

    # --- Model, Optimizer, Scheduler ---
    projector = None

    projector = Projector(
        in_dims=64,
        out_dims=out_dims,
        num_inner_layers=config.num_inner_layers,
        hidden_dims_scale=config.hidden_dims_scale,
    ).to(device)
    encoder = Music2Latent().to(device)
    encoder.eval()  # Encoder is always frozen

    opt = torch.optim.AdamW(
        projector.parameters(),
        lr=config.learning_rate,
        betas=(config.adams_beta1, 0.999),
        eps=config.adams_epsilon,
        amsgrad=True,
    )

    num_steps = 2000
    base_lr = 1.0e-8
    max_lr = 3.0
    # mult = (max_lr / base_lr) ** (1/num_steps)
    def scale(x):
        # return x**4
        return (float(x)/num_steps)**4

    # the scaling fn used by fast.ai:
    #  https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    def fastaiscale(x):
        mult = num_steps ** (1.0/num_steps)

        prod = 1
        for i in range(x):
            prod *= mult

        return prod / num_steps

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        opt, base_lr=base_lr, max_lr=max_lr, step_size_up=num_steps,
        scale_mode='iterations',
        scale_fn=fastaiscale
    )

    loss_fn = get_loss_fn(config.loss)

    # data loaders
    BATCH_SIZE = config.batch_size
    NUM_WORKERS = 4
    ORDERING = OrderOption.QUASI_RANDOM

    PIPELINES = {
        "stems": [NDArrayDecoder(), ToTensor(), ToDevice(device)],
        "mix": [NDArrayDecoder(), ToTensor(), ToDevice(device)],
    }

    train_dl = Loader(
        "/scratch/users/nshaheed/mtg-jamendo-ffcv-latents-100.beton",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        order=ORDERING,
        pipelines=PIPELINES,
        os_cache=False,  # can't cache bc it's too big to fit in memory
        drop_last=True,
    )
    val_dl = Loader(
        "/scratch/users/nshaheed/mtg-jamendo-ffcv-latents-val-50.beton",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        order=OrderOption.SEQUENTIAL,
        pipelines=PIPELINES,
        indices=np.arange(1000),
        os_cache=False,  # can't cache bc it's too big to fit in memory
        drop_last=True,
    )

    # --- Training Loop ---
    projector.train()

    mixes = None
    step = 0

    # if step >= config.max_steps and not ignore_max_steps:
    #     break
    # training loop
    for stems, mix in tqdm(train_dl):
        # for batch in range(0):
        if step > num_steps:
            break

        opt.zero_grad()

        processing_start = time.time()
        loss_dict, z_sum = calculate_losses(projector, stems, mix, config, step)
        loss = loss_dict["total_loss"]

        # update learning rate:

        loss.backward()

        # clip the gradients to prevent exploding
        # nn.utils.clip_grad_value_(projector.parameters(), clip_value=1.0)
        nn.utils.clip_grad_norm_(projector.parameters(), max_norm=2.0, norm_type=2)

        opt.step()


        # --- Logging ---
        # tbatch.set_postfix(loss=loss.item(), sum_loss=loss_dict['sum_loss'].item())
        logging_start = time.time()
        log_dict = {
            "train/loss": loss.item(),  # Log the loss that was backpropagated
            "train/var_loss": loss_dict["var_loss"].item(),
            "train/inv_loss": loss_dict["inv_loss"].item(),
            "train/cov_loss": loss_dict["cov_loss"].item(),
            "train/y_loss": loss_dict["y_loss"].item(),
            "train/y_mix_loss": loss_dict["y_mix_loss"].item(),
            "train/sum_loss": loss_dict["sum_loss"].detach(),
            "train/recon_loss": loss_dict["recon_loss"].item(),
        }

        log_dict = log_dict | {"lr": opt.param_groups[0]['lr']}
        if scheduler:
            scheduler.step()

        val_log = validate(projector, device, val_dl, encoder, config, step=step)
        log_dict.update(val_log)
        run.log(log_dict)

        step += 1


# TODO wrap in runs
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
    parser.add_argument("--base_lr", type=float, help="min lr (for CyclicLR)")
    parser.add_argument("--max_lr", type=float, help="max lr (for CyclicLR)")
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
    parser.add_argument("--num_stems", type=int, default=8)
    parser.add_argument("--num_chunks", type=int, default=10)
    parser.add_argument(
        "--data_dir", type=str, default="/scratch/users/nshaheed/musdb18"
    )

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_epochs", type=int, default=10)
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
        # config = {
        #     "learning_rate": args.learning_rate,
        #     "adams_beta1": args.adams_beta1,
        #     "adams_epsilon": args.adams_epsilon,
        #     "num_inner_layers": args.num_inner_layers,
        #     "hidden_dims_scale": args.hidden_dims_scale,
        #     "projector_dims": args.projector_dims,
        #     "var_coeff": args.var_coeff,
        #     "inv_coeff": args.inv_coeff,
        #     "cov_coeff": args.cov_coeff,
        #     "batch_size": args.batch_size,
        #     "data_dir": args.data_dir,
        #     "max_steps": args.max_steps,
        #     "load_frac": args.load_frac,
        #     "checkpoint_every": args.checkpoint_every,
        #     "val_every": args.val_every,
        #     "loss": args.loss,
        #     "scheduler": args.scheduler,
        #     "augment": args.augment,
        #     "arch": args.arch,
        #     "kld_warmup": args.kld_warmup,
        #     "num_stems": args.num_stems,
        #     "num_chunks": args.num_chunks,
        # }
        config = {}

        config["learning_rate"] = args.learning_rate
        config["base_lr"] = args.base_lr
        config["max_lr"] = args.max_lr
        config["adams_beta1"] = args.adams_beta1
        config["adams_epsilon"] = args.adams_epsilon
        config["num_inner_layers"] = args.num_inner_layers
        config["hidden_dims_scale"] = args.hidden_dims_scale
        config["projector_dims"] = args.projector_dims
        config["var_coeff"] = args.var_coeff
        config["inv_coeff"] = args.inv_coeff
        config["cov_coeff"] = args.cov_coeff
        config["batch_size"] = args.batch_size
        config["data_dir"] = args.data_dir
        config["max_steps"] = args.max_steps
        config["max_epochs"] = args.max_epochs
        config["load_frac"] = args.load_frac
        config["checkpoint_every"] = args.checkpoint_every
        config["val_every"] = args.val_every
        config["loss"] = args.loss
        config["scheduler"] = args.scheduler
        config["augment"] = args.augment
        config["arch"] = args.arch
        config["kld_warmup"] = args.kld_warmup
        config["num_stems"] = args.num_stems
        config["num_chunks"] = args.num_chunks

        config = vars(args)

        with wandb.init(config=config, group="lr_range") as run:
            train(
                run, run.config, ignore_max_steps=args.ignore_max_steps, test=args.test
            )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    main()
