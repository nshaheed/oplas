# train_sweep.py

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
from ffcv.reader import Reader

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


def log_gradients(model, run):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            
            # per-layer stats
            run.log({
                f"grad_norm/{name}": grad.norm().item(),
                f"grad_mean/{name}": grad.mean().item(),
                f"grad_max/{name}": grad.abs().max().item(),
            }, commit=False)

            # accumulate for global norm
            total_norm += grad.norm().item() ** 2

    # log global gradient norm (L2 across all params)
    total_norm = total_norm ** 0.5
    run.log({"grad_norm/global": total_norm}, commit=False)


def log_adaptive_lr(model, optimizer, run):
    for i, group in enumerate(optimizer.param_groups):
        base_lr = group["lr"]  # the scheduled/base learning rate

        for name, p in model.named_parameters():
            if p in optimizer.state:
                state = optimizer.state[p]
                if "exp_avg_sq" in state:  # only exists after a few steps
                    v_hat = state["exp_avg_sq"].sqrt().add_(group["eps"])

                    eff_lr_tensor = base_lr / v_hat
                    eff_lr_mean = eff_lr_tensor.mean().item()
                    eff_lr_max = eff_lr_tensor.max().item()
                    eff_lr_min = eff_lr_tensor.min().item()

                    run.log({
                        f"eff_lr_mean/{name}": eff_lr_mean,
                        f"eff_lr_max/{name}": eff_lr_max,
                        f"eff_lr_min/{name}": eff_lr_min,
                    }, commit=False)



def calculate_losses(projector, ys, y_mix, config, step=0):
    """
    Pass data through projector and calculate losses
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
    loss_fn = get_loss_fn(config.loss)
    vicreg_loss = vicreg_loss_fn(z_sum, z_mix)
    y_loss = loss_fn(ys_tensor, y_hats)
    y_mix_loss = loss_fn(y_mix, y_hat_mix)
    sum_loss = loss_fn(y_mix, y_sum)

    # kld_loss = torch.tensor(0, device=z_mix.device)
    kld_loss = 0
    if config.arch == "vae":
        mu, logvar = params_stems
        kld_loss = kld(mu, logvar, step, config.kld_warmup)

    var_loss = config.var_coeff * vicreg_loss["var_loss"]
    inv_loss = loss_fn(z_mix, z_sum)
    cov_loss = config.cov_coeff * vicreg_loss["cov_loss"]

    total_loss = (
        var_loss + inv_loss + cov_loss + y_loss + y_mix_loss + sum_loss + kld_loss
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
        "kld_loss": kld_loss,
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

def make_examples(ys, y_sum, model, section="train", num_examples=3):
    stems_latent = ys[:num_examples]

    mix_latent = y_sum.permute(0,2,1)[:num_examples]

    examples = {}

    for i in range(num_examples):
        mix = model.decode(mix_latent[i])
    
        stems_mixdown = None
        for j in range(stems_latent.shape[1]):
            result = model.decode(stems_latent[i,j,:,:])
            
            if stems_mixdown is None:
                stems_mixdown = result
            else:
                stems_mixdown += result

        

            # stems = model.decode(stems_latent)

        examples[f"{section}/{i}/z_mix"] = wandb.Audio(
            mix[0],
            caption="decoding of audio mixed in the z domain",
            sample_rate=44100,
        )
        examples[f"{section}/{i}/z_target"] = wandb.Audio(
            stems_mixdown[0],
            caption="stems decoded separately and mixed together",
            sample_rate=44100,
        )

    return examples



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

    if config.arch == "vae":
        log = log | {"val/kld_loss": loss_avg["kld_loss"].item()}

    # actually generate some audio examples
    latent = projector.decode(z_sum.permute(0, 2, 1))  # swap last two dims

    examples = make_examples(stems, latent, encoder, section="val")
    log = log | examples

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

    opt = torch.optim.AdamW(
        projector.parameters(),
        lr=config.learning_rate,
        betas=(config.adams_beta1, 0.999),
        eps=config.adams_epsilon,
        amsgrad=True,
    )

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
    elif config.run_from:
        print(f"Beginning run from {config.run_from}")

        try:
            artifact = run.use_artifact(f"model-ckpt-{config.run_from}:latest")
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
    # stems_dataset = StemChunkStream(
    #     data_dir=config.data_dir,
    #     # data_dir="/scratch/users/nshaheed/mtg-jamendo-wav/",
    #     load_frac=config.load_frac,
    #     augment=config.augment,
    # )
    # mtg_dataset = MTGJamendoStreamSingle(
    #     data_dir="/scratch/users/nshaheed/mtg-jamendo-wav/",
    #     augment=config.augment,
    #     load_frac=0.01,
    #     chunk_size=2**16,
    # )
    # mtg_dataset = MTGJamendoCache(num_chunks=config.num_chunks)
    # train_dataset, val_dataset = torch.utils.data.random_split(mtg_dataset, [0.8, 0.2])

    # train_dataset = RandomMixDataset([stems_dataset, mtg_dataset])
    # val_dataset = StemChunkStream(
    #     data_dir=config.data_dir, subset="test", load_frac=config.load_frac
    # )

    if True:
        # # for debugging on sh_dev
        # train_dl = DataLoader(  # do 12 workers?
        #     mtg_dataset,
        #     batch_size=config.batch_size,
        #     num_workers=0,
        #     pin_memory=True,
        #     drop_last=True,
        #     shuffle=True,
        # )
        # val_dl = DataLoader(
        #     mtg_dataset,
        #     batch_size=config.batch_size,
        #     num_workers=0,
        #     pin_memory=True,
        #     drop_last=True,
        #     shuffle=True,
        # )

        BATCH_SIZE = config.batch_size
        NUM_WORKERS = 4
        ORDERING = OrderOption.QUASI_RANDOM

        # PIPELINES = {
        #     "audio": [NDArrayDecoder(), ToTensor(), ToDevice(device)],
        # }
        PIPELINES = {
            "stems": [NDArrayDecoder(), ToTensor(), ToDevice(device)],
            "mix": [NDArrayDecoder(), ToTensor(), ToDevice(device)],
        }

        train_file = f"/scratch/users/nshaheed/mtg-jamendo-ffcv-latents-train-{config.num_stems}.beton"
        indices = None
        if config.load_frac < 1.0:
            reader = Reader(train_file) # read the train file to get it's length
            indices = np.arange(int(config.load_frac * reader.num_samples))

        train_dl = Loader(
            train_file,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            order=ORDERING,
            pipelines=PIPELINES,
            indices=indices,
            os_cache=False,  # can't cache bc it's too big to fit in memory
            drop_last=True,
        )

        val_file = f"/scratch/users/nshaheed/mtg-jamendo-ffcv-latents-val-{config.num_stems}.beton"
        val_load_frac = config.val_load_frac if "val_load_frac" in config else 1.0
        if val_load_frac < 1.0:
            reader = Reader(val_file) # read the train file to get it's length
            indices = np.arange(int(config.val_load_frac * reader.num_samples))

        val_dl = Loader(
            val_file,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            order=ORDERING,
            pipelines=PIPELINES,
            indices=indices,
            os_cache=False,  # can't cache bc it's too big to fit in memory
            drop_last=True,
        )
    else:
        train_dl = DataLoader(  # do 12 workers?
            train_dataset,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
        val_dl = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    scheduler = get_scheduler(opt, config, train_dl)
    # --- Training Loop ---
    projector.train()

    # repeat one batch infinitely
    # if test:
    #     tbatch = tqdm(
    #         repeat(next(iter(train_dl))),
    #         initial=start_step,
    #         total=config.max_steps,
    #         unit="batch",
    #         desc="training",
    #     )
    #     # tbatch = repeat(next(iter(tbatch)))

    mixes = None

    step = start_step
    # tbatch = tqdm(
    #     initial=step,
    #     total=config.max_steps,
    #     unit="b",
    #     desc="Trn",
    #     smoothing=0,
    # )
    max_epochs = config.max_epochs
    epochs = tqdm(range(max_epochs), desc="epoch", smoothing=0)
    step_test = 0
    step = 0
    for i in epochs:
        tbatch = tqdm(train_dl, desc="steps", smoothing=0, leave=False)
        dataload_start = time.time()

        # training loop
        for stems, mix in tbatch:
            # for batch in range(0):
            # if tbatch.n > 50:
            #     break

            # print(f"Data loading time: {time.time() - dataload_time:.05f}s")
            dataload_end = time.time() - dataload_start
            # step = tbatch.n
            # step += 1
            # step = epochs.n * len(train_dl) + tbatch.n
            
            opt.zero_grad()

            # # resize batch so that it has shape [batch, num_stems, latent, time]
            # b, t = batch.shape
            # s = config.num_stems
            # batch = batch.view(b // s, s, t)

            # # augment with effects
            # if config.augment:
            #     batch = augment_effects(batch, sample_rate=44100)

            # encoding_start = time.time()
            # with torch.no_grad():
            #     if test and mixes is None:
            #         mixes = mix_single(batch, encoder, static_mix=False, debug=False)
            #     if not test:
            #         mixes = mix_single(batch, encoder, static_mix=False, debug=False)

            # print(f"Encoding time:     {time.time() - encoding_time:.05f}s")
            # encoding_end = time.time() - encoding_start

            processing_start = time.time()
            loss_dict, z_sum = calculate_losses(projector, stems, mix, config, step)
            loss = loss_dict["total_loss"]
            loss.backward()

            # log gradients every 100 steps
            if step % 1000 == 0:
                log_gradients(projector, run)


            # clip the gradients to prevent exploding
            nn.utils.clip_grad_norm_(projector.parameters(), max_norm=2.0, norm_type=2)


            opt.step()
            if scheduler and config.scheduler != "reduce_on_plateau":
                scheduler.step()

            if step % 1000 == 0:
                log_adaptive_lr(projector, opt, run)


            processing_end = time.time() - processing_start

            # print(f"Processing time:   {time.time() - processing_time:.05f}s")

            # --- Logging ---
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

            # update learning rate:
            log_dict = log_dict | {"lr": opt.param_groups[0]['lr']}

            if config.arch == "vae":
                log_dict = log_dict | {"train/kld_loss": loss_dict["kld_loss"].item()}

            if scheduler:
                log_dict = log_dict | {
                    "train/learning_rate": scheduler.get_last_lr()[0]
                }

            logging_end = time.time() - logging_start
            epochs.set_postfix(
                d=dataload_end, p=processing_end, l=logging_end
            )
            tbatch.set_postfix(loss=loss.item(), sum=loss_dict["sum_loss"].item())
            tbatch.update(1)

            if step % config.checkpoint_every == 0 and step != 0:
                # checkpoint and generate some examples
                latent = projector.decode(z_sum.permute(0, 2, 1))  # swap last two dims
                # latent = latent.permute(0, 2, 1)  # have to swap it back
                examples = make_examples(stems, latent, encoder, section="train")

                log_dict = log_dict | examples

                checkpoint_path = save_model(
                    projector, step, opt, scheduler, loss, run.id
                )
                # Log checkpoint as a versioned artifact
                artifact = wandb.Artifact(f"model-ckpt-{run.id}", type="model")
                artifact.add_file(local_path=checkpoint_path, name="model.pt")
                run.log_artifact(artifact)

            dataload_start = time.time()
            
            if step % (config.val_every * 1) == 0 and not test:
                val_log = validate(projector, device, val_dl, encoder, config, step=step)
                log_dict.update(val_log)
    
                checkpoint_path = save_model(projector, step, opt, scheduler, loss, run.id)
                # Log checkpoint as a versioned artifact
                artifact = wandb.Artifact(f"model-ckpt-{run.id}", type="model")
                artifact.add_file(local_path=checkpoint_path, name="model.pt")
                run.log_artifact(artifact)

                # ReduceLROnPlateau can use validation loss for this
                if config.scheduler == 'reduce_on_plateau':
                    scheduler.step(val_log['val/loss'])

            run.log(log_dict)
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
    parser.add_argument(
        "--run_from",
        type=str,
        default=None,
        help="W&B run ID to begin training a new model with.",
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
    parser.add_argument("--val_load_frac", type=float, default=1.0)
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

    elif args.run_from: # new run using existing model as starting point
        config = vars(args)
        with wandb.init(config=config) as run:
            train(
                run, run.config, ignore_max_steps=args.ignore_max_steps, test=args.test
            )

    # --- Mode 3: Single run ---
    else:
        config = vars(args)

        with wandb.init(config=config) as run:
            train(
                run, run.config, ignore_max_steps=args.ignore_max_steps, test=args.test
            )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    main()
