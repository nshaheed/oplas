# train_sweep.py

import argparse
from itertools import islice
from pathlib import Path
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
# Assuming your custom modules are in the python path
from oplas.data import StemChunk
from oplas.losses import vicreg_loss_fn
from oplas.mixing import mix_and_encode
from oplas.models import Music2Latent, Projector

# Best practice: define functions at the top level
def save_model(model, step, optimizer, scheduler, loss, run_id, save_dir="./checkpoints"):
    """Saves the model checkpoint in a run-specific directory."""
    run_dir = Path(save_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / f"checkpoint_step_{step}.pt"

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None, # Save scheduler state
        'loss': loss,
    }, save_path)
    return save_path

@torch.no_grad()
def validate(projector, device, val_dl, encoder, config):
    """Validation function, slightly simplified for clarity."""
    projector.eval() # Set model to evaluation mode
    vbatch_iter = iter(val_dl)
    try:
        batch = next(vbatch_iter).to(device)
    except StopIteration:
        print("Validation dataloader is empty.")
        return {}

    # --- Your validation logic remains the same ---
    mixes = mix_and_encode(batch, encoder)
    y_mix = mixes["y_mix"]
    z_sum = None

    z_mix, y_hat_mix = projector(y_mix.permute(0, 2, 1))
    z_mix = z_mix.permute(0, 2, 1)
    y_hat_mix = y_hat_mix.permute(0, 2, 1)

    ys = mixes["ys"]
    y_hats = []
    
    # This loop is inefficient. Let's vectorize it.
    # Original loop processed each time step independently. We can do it all at once.
    # New vectorized approach:
    ys_tensor = torch.stack(ys, dim=1).to(device) # [B, S, D, T] S=num_stems
    B, S, D, T = ys_tensor.shape
    ys_permuted = ys_tensor.permute(0, 1, 3, 2).reshape(B * S * T, D) # Reshape for projector
    
    z_stems_flat, y_hats_flat = projector(ys_permuted)
    
    z_stems = z_stems_flat.reshape(B, S, T, D).permute(0, 1, 3, 2) # Reshape back
    y_hats_tensor = y_hats_flat.reshape(B, S, T, D).permute(0, 1, 3, 2)
    
    z_sum = torch.sum(z_stems, dim=1) # Sum over stems dimension

    mseloss = nn.MSELoss()
    vicreg_loss = vicreg_loss_fn(z_sum, z_mix)
    y_loss = mseloss(ys_tensor, y_hats_tensor)
    y_mix_loss = mseloss(y_mix, y_hat_mix)

    var_loss = config.var_coeff * vicreg_loss["var_loss"]
    inv_loss = config.inv_coeff * vicreg_loss["inv_loss"]
    cov_loss = config.cov_coeff * vicreg_loss["cov_loss"]

    loss = var_loss + inv_loss + cov_loss + y_loss + y_mix_loss
    
    log = {
        "val/loss": loss.detach(),
        "val/var_loss": var_loss.detach(),
        "val/inv_loss": inv_loss.detach(),
        "val/cov_loss": cov_loss.detach(),
        "val/y_loss": y_loss.detach(),
        "val/y_mix_loss": y_mix_loss.detach(),
    }
    
    projector.train() # Set model back to training mode
    return log

def train(args):
    """Main training function wrapped for W&B sweeps."""
    # Initialize a new wandb run
    run = wandb.init()
    config = wandb.config # Get hyperparams from the sweep

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # --- Data Loading ---
    train_dataset = StemChunk(data_dir=args.data_dir, load_frac=config.load_frac)
    val_dataset = StemChunk(data_dir=args.data_dir, subset="test", load_frac=config.load_frac)
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)

    # --- Model, Optimizer, Scheduler ---
    projector = Projector(
        in_dims=64, out_dims=64, num_inner_layers=config.num_inner_layers, hidden_dims_scale=config.hidden_dims_scale
    ).to(device)
    encoder = Music2Latent().to(device)
    encoder.eval() # Encoder is always frozen

    opt = torch.optim.Adam(projector.parameters(), lr=config.learning_rate)
    # TODO use fixed learning rate for now
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.learning_rate, total_steps=config.max_steps
    )
    mseloss = nn.MSELoss()
    
    start_step = 0
    # --- W&B Resuming Logic ---
    if wandb.run.resumed:
        print("Resuming run from a checkpoint...")
        # wandb.restore() downloads the file and returns a file path
        # It's safer to restore the most recent checkpoint. We'll log it as an artifact.
        try:
            artifact = run.use_artifact(f'model-ckpt-{run.id}:latest')
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


    # --- Training Loop ---
    projector.train()
    step_iterator = islice(iter(train_dl), start_step, config.max_steps)
    tbatch = tqdm(step_iterator, initial=start_step, total=config.max_steps, unit="batch", desc="training")
    
    for step, batch in enumerate(tbatch, start=start_step):
        if step >= config.max_steps:
            break

        batch = batch.to(device)
        opt.zero_grad()

        with torch.no_grad():
            mixes = mix_and_encode(batch, encoder)
            y_mix = mixes["y_mix"].to(torch.float32)

        # Vectorized projection logic (more efficient)
        z_mix, y_hat_mix = projector(y_mix.permute(0, 2, 1))
        z_mix = z_mix.permute(0, 2, 1)
        y_hat_mix = y_hat_mix.permute(0, 2, 1)

        # TODO validate
        ys_tensor = torch.stack(mixes["ys"], dim=1).to(torch.float32)
        B, S, D, T = ys_tensor.shape
        ys_permuted = ys_tensor.permute(0, 1, 3, 2).reshape(B * S * T, D)
        z_stems_flat, y_hats_flat = projector(ys_permuted)
        z_stems = z_stems_flat.reshape(B, S, T, D).permute(0, 1, 3, 2)
        y_hats_tensor = y_hats_flat.reshape(B, S, T, D).permute(0, 1, 3, 2)
        z_sum = torch.sum(z_stems, dim=1)

        # --- Loss Calculation ---
        vicreg_loss = vicreg_loss_fn(z_sum, z_mix)
        y_loss = mseloss(ys_tensor, y_hats_tensor)
        y_mix_loss = mseloss(y_mix, y_hat_mix)

        var_loss = config.var_coeff * vicreg_loss["var_loss"]
        inv_loss = config.inv_coeff * vicreg_loss["inv_loss"]
        cov_loss = config.cov_coeff * vicreg_loss["cov_loss"]
        
        loss = var_loss + inv_loss + cov_loss + y_loss + y_mix_loss
        
        loss.backward()
        opt.step()
        scheduler.step()

        # --- Logging ---
        tbatch.set_postfix(loss=loss.item(), inv_loss=inv_loss.item())
        log_dict = {
            "train/loss": loss.item(),
            "train/var_loss": var_loss.item(),
            "train/inv_loss": inv_loss.item(),
            "train/cov_loss": cov_loss.item(),
            "train/y_loss": y_loss.item(),
            "train/y_mix_loss": y_mix_loss.item(),
            "learning_rate": scheduler.get_last_lr()[0]
        }
        
        if step % config.val_every == 0:
            val_log = validate(projector, device, val_dl, encoder, config)
            log_dict.update(val_log)

        run.log(log_dict, step=step)

        if step % config.checkpoint_every == 0 and step > 0:
            checkpoint_path = save_model(projector, step, opt, scheduler, loss, run.id)
            # Log checkpoint as a versioned artifact
            artifact = wandb.Artifact(f'model-ckpt-{run.id}', type='model')
            artifact.add_file(local_path=checkpoint_path, name="model.pt")
            run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a W&B sweep agent for the oplas model.")
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="/scratch/users/nshaheed/musdb18",
        help="Path to musdb18 files (does not change across runs)",
    )
    args = parser.parse_args()
    
    train(args)
