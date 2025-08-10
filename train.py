import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import islice

import wandb
from oplas.data import StemDataset2, StemDataset, StemChunkStream
from oplas.losses import vicreg_loss_fn
from oplas.mixing import mix_and_encode
from oplas.models import Music2Latent, Projector, VGGishEncoder


def save_model(model, save_dir="./", model_path="projector.pt", suffix=""):
    dir = Path(save_dir)
    dir.mkdir(parents=True, exist_ok=True)  # make save dir if needed
    save_path = dir / model_path.replace(".pt", f"{suffix}.pt")
    torch.save(model, save_path)

@torch.no_grad()
def validate(projector, device, run, val_dl, model):
    vbatch = tqdm(islice(val_dl, 0, 50), unit="batch", total=100)
    for step, batch in enumerate(vbatch):
        vbatch.set_description("validating")

        batch = batch.to(device)

        mixes = mix_and_encode(batch, encoder)
        y_mix = mixes["y_mix"]
        # zs is a list of the encoded stems
        zs = []

        z_sum = None

        # project y_mix?
        z_mix_chunks = []
        for i in range(y_mix.shape[-1]):
            # need to process each latent value independently in the audio tracks
            z_mix_chunk, y_hat_chunk = projector(y_mix[:, :, i])
            z_mix_chunks.append(z_mix_chunk)

        z_mix = torch.stack(z_mix_chunks, -1)

        # go through each stem, project it, and then recombine the projection into z_sum
        for y in mixes["ys"]:
            z_chunks = []
            for i in range(
                    y.shape[-1]
            ):  # need to take each latent chunk independently
                z_chunk, _ = projector(y[:, :, i])
                # breakpoint()
                z_chunks.append(z_chunk)

            z = torch.stack(z_chunks, -1)
            z_sum = z if z_sum is None else z + z_sum
            zs.append(z)

        mix_loss = mseloss(z_sum, z_mix)
        vicreg_loss = vicreg_loss_fn(z_sum, z_mix)

        loss = (
            mix_loss
            + vicreg_loss["var_loss"]
            + vicreg_loss["inv_loss"]
            + vicreg_loss["cov_loss"]
        )

        # generate some media
        if step == 0:
            latent = projector.decode(z_sum.permute(0,2,1)) # swap last two dims
            latent = latent.permute(0,2,1) # have to swap it back
            # latent = latent.view(latent.shape[0], -1)
            audio = model.decode(latent).cpu()
            run.log({
                "val/z_mix0": wandb.Audio(audio[0], caption="decoding of audio mixed in the z domain", sample_rate=44100),
                "val/orig0": wandb.Audio(batch[0,0,:,0].cpu(), caption="original mix", sample_rate=44100),
                "val/z_mix1": wandb.Audio(audio[1], caption="decoding of audio mixed in the z domain", sample_rate=44100),
                "val/orig1": wandb.Audio(batch[1,0,:,0].cpu(), caption="original mix", sample_rate=44100),
                "val/z_mix2": wandb.Audio(audio[2], caption="decoding of audio mixed in the z domain", sample_rate=44100),
                "val/orig2": wandb.Audio(batch[2,0,:,0].cpu(), caption="original mix", sample_rate=44100)
                })



        vbatch.set_postfix(loss=loss.item(), mix_loss=mix_loss.item())
        run.log(
            {
                "val/loss": loss.detach(),
                "val/mix_loss": mix_loss.detach(),
                "val/var_loss": vicreg_loss["var_loss"].detach(),
                "val/inv_loss": vicreg_loss["inv_loss"].detach(),
                "val/cov_loss": vicreg_loss["cov_loss"].detach(),
            }
        )


# TODO add train/test split and validation

parser = argparse.ArgumentParser(description="Train or run the music2latent model.")
parser.add_argument(
    "-d",
    "--data-dir",
    type=str,
    default="/scratch/users/nshaheed/musdb18",
    help="path to musdb18 files",
)
parser.add_argument(
    "-t",
    "--test",
    action="store_true",
    help="do test run with subset of data and smaller batch size",
)
args = parser.parse_args()

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")

seed = 42
torch.manual_seed(seed)


wandb.login()

project = "oplas-latent-mix-projection"
config = {
    "max_epochs": 40,
    "max_lr": 0.002,
    "test": args.test,
    "batch_size": 10 if args.test else 300,
    "load_frac": 0.01 if args.test else 1.0,
}

projector = Projector(in_dims=64, out_dims=64).to(device)

max_epochs = config["max_epochs"]
lossinfo_every, viz_demo_every = 20, 1000  # in units of steps
checkpoint_every = 200
val_every = 100
max_lr = 0.002
batch_size = config["batch_size"]
dataset_size = 100

# loading the datasets
load_frac = config["load_frac"]

train_dataset = StemChunkStream(data_dir=args.data_dir, load_frac=load_frac, debug=False)
val_dataset = StemChunkStream(
    data_dir=args.data_dir, subset="test", load_frac=load_frac*2
)

train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
val_dl = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)

max_steps = 100000

# optimizer and learning rate scheduler
opt = torch.optim.Adam([*projector.parameters()], lr=5e-4)
total_steps = dataset_size // batch_size * max_epochs
print("total_steps =", total_steps)  # for when I'm checking wandb
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=max_lr, total_steps=max_steps
)
mseloss = nn.MSELoss()

# encoder is music2latent (non-streaming for now)
encoder = Music2Latent()
# encoder = VGGishEncoder()
# encoder = CLAPEncoder()

# training loop
with wandb.init(project=project, config=config) as run:
    epoch, step = 0, 0
    tbatch = tqdm(train_dl, total=max_steps, unit="batch")
    for step, batch in enumerate(tbatch):
        # breakpoint()
        if step >= max_steps:
            break

        batch = batch.to(device)
        opt.zero_grad()
        log_dict = {}
        tbatch.set_description('training')

        with torch.no_grad():  # encoder is frozen
            # mix = mix_stems(batch, static_mix=False)
            mixes = mix_and_encode(batch, encoder)
            y_mix = mixes["y_mix"]

        # zs is a list of the encoded stems
        zs = []
        z_sum = None

        # project y_mix?
        z_mix_chunks = []
        for i in range(y_mix.shape[-1]):
            # need to process each latent value independently in the audio tracks
            z_mix_chunk, y_hat_chunk = projector(y_mix[:, :, i])
            z_mix_chunks.append(z_mix_chunk)

        z_mix = torch.stack(z_mix_chunks, -1)

        # go through each stem, project it, and then recombine the projection into z_sum
        for y in mixes["ys"]:
            z_chunks = []
            for i in range(
                y.shape[-1]
            ):  # need to take each latent chunk independently
                z_chunk, _ = projector(y[:, :, i])
                # breakpoint()
                z_chunks.append(z_chunk)

            z = torch.stack(z_chunks, -1)
            z_sum = z if z_sum is None else z + z_sum
            zs.append(z)

        mix_loss = mseloss(z_sum, z_mix)
        vicreg_loss = vicreg_loss_fn(z_sum, z_mix)

        loss = (
            mix_loss
            + vicreg_loss["var_loss"]
            + vicreg_loss["inv_loss"]
            + vicreg_loss["cov_loss"]
        )
        tbatch.set_postfix(loss=loss.item(), mix_loss=mix_loss.item())
        run.log(
            {
                "train/loss": loss.detach(),
                "train/mix_loss": mix_loss.detach(),
                "train/var_loss": vicreg_loss["var_loss"].detach(),
                "train/inv_loss": vicreg_loss["inv_loss"].detach(),
                "train/cov_loss": vicreg_loss["cov_loss"].detach(),
            }
        )

        if step % checkpoint_every == 0:
            save_model(
                projector,
                save_dir="./checkpoints",
                suffix=f"_{run.name}_{step}",
            )

        if step % val_every == 0:
            validate(projector, device, run, val_dl, encoder)

        loss.backward()
        opt.step()
        step += 1

    save_model(projector, save_dir="./checkpoints", suffix=f"_{run.name}_{step}")

print("done")
