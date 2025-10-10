import os
import math
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder
from ffcv.transforms import ToTensor, ToDevice
from glob import glob
from torch.utils.data import Dataset
from tqdm import tqdm

from oplas.models import Music2Latent
from oplas.mixing import mix_and_encode, mix_single

CHUNK_SIZE = 2**18  # Number of samples per chunk (262144)
LATENT_SIZE = 64
LATENT_LEN = 63 # num of latents per iter




class MixingDataset(Dataset):
    def __init__(self, loader, num_samples, num_stems, encoder):
        self.loader = loader
        self.num_samples = num_samples
        self.num_stems = num_stems
        self._iter = iter(self.loader)

        self.yss = []
        self.y_mixes = []
        
        # load everything
        for i in tqdm(range(num_samples)):
            try:
                (batch,) = next(self._iter)
            except StopIteration:
                self._iter = iter(self.loader)
                (batch,) = next(self._iter)

            batch = batch.unsqueeze(0)
            mixes = mix_single(batch, encoder, static_mix=False, debug=False)

            ys = mixes['ys'].squeeze(0).cpu().numpy().astype('float32')
            y_mix = mixes['y_mix'].squeeze(0).cpu().numpy().astype('float32')

            self.yss.append(ys)
            self.y_mixes.append(y_mix)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.yss[idx], self.y_mixes[idx]

        # zer = np.zeros((LATENT_SIZE,LATENT_LEN))
        # print(f'{zer.dtype=}')
        # print(f'{type(zer)=}')
        # yield torch.zeros((LATENT_SIZE,LATENT_LEN)).numpy()

        try:
            (batch,) = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            (batch,) = next(self._iter)

        print(f'getting item! {idx=}')

        
        batch = batch.unsqueeze(0)
        mixes = mix_single(batch, encoder, static_mix=False, debug=False)

        ys = mixes['ys'].squeeze(0).cpu().numpy().astype('float32')
        y_mix = mixes['y_mix'].squeeze(0).cpu().numpy().astype('float32')
        # print(f"{ys.shape=}")
        # print(f"{y_mix.shape=}")
        # print(f"{type(y_mix)=}")
        # print(f'{y_mix.dtype=}')
        # print(f'{torch.zeros((LATENT_SIZE,LATENT_LEN)).numpy().shape == y_mix.shape=}')
                
        return(ys, y_mix)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_path", type=str, required=True, help="Path to the orig audio ffcv dataset (.beton)"
    )
    parser.add_argument(
        "--out_path", type=str, required=True, help="Path to save ffcv latent dataset (.beton)"
    )

    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    # parser.add_argument("--batch_size", type=int, default=32, help="Number of samples to generate")
    parser.add_argument("--num_stems", type=int, default=16, help="Number of stems to mix")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of stems to mix")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_WORKERS=args.num_workers
    BATCH_SIZE=args.num_stems
    ORDERING = OrderOption.RANDOM
    PIPELINES = {
        "audio": [NDArrayDecoder(), ToTensor(), ToDevice(device)],
    }
    loader = Loader(
        args.in_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pipelines=PIPELINES,
        os_cache=False,  # can't cache bc it's too big to fit in memory
        drop_last=True,
    )

    encoder = Music2Latent().to(device)
    encoder.eval()  # Encoder is always frozen

    mixing_dataset = MixingDataset(loader, args.num_samples, BATCH_SIZE, encoder=encoder)



    print(f'{args.num_stems=}')
    print(f'{(args.num_stems,LATENT_SIZE,LATENT_LEN)=}')
    print(f'{(LATENT_SIZE,LATENT_LEN)=}')

    writer = DatasetWriter(
        f"{args.out_path}.beton",
        {
            "stems": NDArrayField(shape=(args.num_stems,LATENT_SIZE,LATENT_LEN), dtype=np.dtype("float32")),
            "mix": NDArrayField(shape=(LATENT_SIZE,LATENT_LEN), dtype=np.dtype("float32"))
        },
        num_workers=args.num_workers,
    )

    # gen = latent_generator(loader, encoder, args.num_stems)

    # writer.from_indexed_dataset(gen)

    writer.from_indexed_dataset(mixing_dataset)

    # uv run ffcv_prebaked.py --in_path $SCRATCH/mtg-jamendo-ffcv-train.beton --out_path $SCRATCH/mtg-jamendo-ffcv-latents --num_workers 4 --num_stems 16 --num_samples 70000

    # tqdm on while loop, exit when # steps reaches num_samples
    
