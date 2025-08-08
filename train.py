import torch
from torch import nn

import numpy as np
from tqdm import tqdm
# import torchdata
from torch.utils.data import DataLoader

import argparse

from oplas.models import Projector, Music2Latent, VGGishEncoder
from oplas.data import StemDataset
from oplas.mixing import mix_and_encode
from oplas.losses import vicreg_loss_fn

parser = argparse.ArgumentParser(description="Train or run the music2latent model.")
parser.add_argument("-d", "--data-dir", type=str, default="/scratch/users/nshaheed/musdb18", help="path to musdb18 files")
parser.add_argument("-t", "--test", action='store_true', help="do test run with subset of data and smaller batch size")
args = parser.parse_args()

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')

seed = 42
torch.manual_seed(seed)

# TODO put things on gpuo
projector = Projector(in_dims=64,out_dims=64).to(device)

max_epochs = 40
lossinfo_every, viz_demo_every = 20, 1000   # in units of steps
checkpoint_every = 10000
max_lr = 0.002
# batch_size = 1024 # TODO undo this
batch_size = 10 if args.test else 1024
# batch_size = 10

# loading the datasets

load_frac = 0.01 if args.test else 1.0

train_dataset = StemDataset(data_dir=args.data_dir, preload=True, load_frac=load_frac) 
val_dataset = StemDataset(data_dir=args.data_dir, subset='test', preload=True, load_frac=load_frac)

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# optimizer and learning rate scheduler
opt = torch.optim.Adam([*projector.parameters()], lr=5e-4) 
total_steps = len(train_dataset)//batch_size * max_epochs
print("total_steps =",total_steps)  # for when I'm checking wandb
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total_steps)
mseloss = nn.MSELoss()

# encoder is music2latent (non-streaming for now)
encoder = Music2Latent()
# encoder = VGGishEncoder()
# encoder = CLAPEncoder()

# training loop
epoch, step = 0, 0
while (epoch < max_epochs) or (max_epochs < 0):  # training loop
    with tqdm(train_dl, unit="batch") as tepoch:
        for batch in tepoch:   # train
            batch = batch.to(device)
            opt.zero_grad()
            log_dict = {}

            with torch.no_grad(): # encoder is frozen
                # mix = mix_stems(batch, static_mix=False)
                mixes = mix_and_encode(batch, encoder)
                y_mix = mixes['y_mix']

            # zs is a list of the encoded stems
            zs = []
            z_sum = None

            # project y_mix?
            z_mix_chunks = []
            for i in range(y_mix.shape[-1]):
                # need to process each latent value independently in the audio tracks
                z_mix_chunk, y_hat_chunk = projector(y_mix[:,:,i])
                z_mix_chunks.append(z_mix_chunk)

            z_mix = torch.stack(z_mix_chunks, -1)
            
            # go through each stem, project it, and then recombine the projection into z_sum
            for y in mixes['ys']:
                z_chunks = []
                for i in range(y.shape[-1]): # need to take each latent chunk independently
                    z_chunk, _ = projector(y[:,:,i])
                    # breakpoint()
                    z_chunks.append(z_chunk)

                z = torch.stack(z_chunks, -1)
                z_sum = z if z_sum is None else z + z_sum
                zs.append(z)

            mix_loss = mseloss(z_sum, z_mix)
            vicreg_loss = vicreg_loss_fn(z_sum, z_mix)

            loss = mix_loss + vicreg_loss['var_loss'] + vicreg_loss['inv_loss'] + vicreg_loss['cov_loss']
            tepoch.set_postfix(loss=loss.item(), mix_loss=mix_loss.item())
            
            loss.backward()
            opt.step()
            
            # breakpoint()

            # stems, faders, train_iter = get_stems_faders(batch, train_iter, train_dl, debug=debug)

print('done')
