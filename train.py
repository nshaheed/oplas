import torch

import numpy as np
from tqdm import tqdm
# import torchdata
from torch.utils.data import DataLoader

from oplas.models import Projector
from oplas.data import StemDataset
from oplas.mixing import mix_stems

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

seed = 42
torch.manual_seed(seed)

# TODO put things on gpuo
projector = Projector(in_dims=64,out_dims=64).to(device)

max_epochs = 40
lossinfo_every, viz_demo_every = 20, 1000   # in units of steps
checkpoint_every = 10000
max_lr = 0.002
# batch_size = 1024 # TODO undo this
batch_size = 10

# loading the datasets
# TODO change loading amount
train_dataset = StemDataset(data_dir='/scratch/users/nshaheed/musdb18', preload=True, load_frac=0.01) 
val_dataset = StemDataset(data_dir='/scratch/users/nshaheed/musdb18', subset='test', preload=True, load_frac=0.02)

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print('omg')

# optimizer and learning rate scheduler
opt = torch.optim.Adam([*projector.parameters()], lr=5e-4) 
total_steps = len(train_dataset)//batch_size * max_epochs
print("total_steps =",total_steps)  # for when I'm checking wandb
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total_steps)


# training loop
epoch, step = 0, 0
while (epoch < max_epochs) or (max_epochs < 0):  # training loop
    with tqdm(train_dl, unit="batch") as tepoch:
        for batch in tepoch:   # train
            batch = batch.to(device)
            opt.zero_grad()
            log_dict = {}

            breakpoint()
            mix = mix_stems(batch, static_mix=False)

            # stems, faders, train_iter = get_stems_faders(batch, train_iter, train_dl, debug=debug)

print('done')
