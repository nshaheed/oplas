#!/usr/bin/env python3
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from ffcv.loader import Loader
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice

import threading


class BetonInMemory:
    def __init__(self, beton_files, num_workers, device):
        self.pipelines = {
            "stems": [NDArrayDecoder(), ToTensor(), ToDevice(torch.device(device))],
            "mix": [NDArrayDecoder(), ToTensor(), ToDevice(torch.device(device))],
        }

        self.beton_files = beton_files
        self.num_workers = num_workers
        self.iter = None
        self.next_file = 0 # idx of next file

        total = 0
        print("Counting samples...")
        for beton_path in tqdm(beton_files):
            loader = Loader(
                beton_path,
                batch_size=1,
                num_workers=1,
                pipelines=self.pipelines,
                os_cache=False,
                drop_last=False,
            )
            length = len(loader.indices)
            total += length
        self.total_len = total
        print(f'{self.total_len=}')

        # self.stems = []
        # self.mixes = []
        # for beton_file in tqdm(beton_files):
        #     loader = Loader(
        #         beton_file,
        #         batch_size=1,
        #         num_workers=num_workers,
        #         pipelines=self.pipelines,
        #         os_cache=False,
        #         drop_last=False,
        #     )
        #     for batch in tqdm(loader):
        #         stems = batch[0][0].numpy().astype('float32')
        #         mix = batch[1][0].numpy().astype('float32')

        #         self.stems.append(stems)
        #         self.mixes.append(mix)

    def __len__(self):
        # return len(self.mixes)
        return self.total_len

    def __getitem__(self, idx):
        # print('fjdklsfjldkgetitem')
        # while True:
        #     yield (self.stems[idx], self.mixes[idx])
        # generalize this to work w/ multiple loaders

        # initial set up
        # if idx % 13 == 0:
        #     print(idx)
        if self.iter is None:
            loader = Loader(
                self.beton_files[0],
                batch_size=1,
                num_workers=self.num_workers,
                pipelines=self.pipelines,
                os_cache=False,
                drop_last=False,
            )
            self.iter= iter(loader)
            self.next_file = 1

        try:
            batch = next(self.iter)
        except StopIteration:
            print(f'new loader!! {self.beton_files[self.next_file]}')
            loader = Loader(
                self.beton_files[self.next_file],
                batch_size=1,
                num_workers=self.num_workers,
                pipelines=self.pipelines,
                os_cache=False,
                drop_last=False,
            )
            self.iter= iter(loader)
            self.next_file += 1

            batch = next(self.iter)

        stems = batch[0][0].numpy().astype('float32')
        mix = batch[1][0].numpy().astype('float32')
        # print(f'{stems.shape=}, {mix.shape=}')
        return(stems, mix)
        # return (self.stems[idx], self.mixes[idx])


class CombinedBetonDataset:
    """
    Dataset-like wrapper that merges multiple .beton datasets sequentially.
    Provides indexed access compatible with FFCV's DatasetWriter.
    """

    def __init__(self, beton_files, num_workers, batch_size, device, lock):
        self.beton_files = beton_files
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.lock = lock
        self.pipelines = {
            "stems": [NDArrayDecoder(), ToTensor(), ToDevice(self.device)],
            "mix": [NDArrayDecoder(), ToTensor(), ToDevice(self.device)],
        }

        print('DATASETS')
        # Precompute dataset boundaries (to enable indexing)
        self.sample_offsets = []
        self.loaders = []
        total = 0
        print("Indexing datasets...")
        for beton_path in tqdm(beton_files):
            loader = Loader(
                beton_path,
                batch_size=1,
                num_workers=1,
                pipelines=self.pipelines,
                os_cache=False,
                drop_last=False,
            )
            length = len(loader.indices)
            self.sample_offsets.append((total, total + length, beton_path, len(self.loaders)))
            total += length
            self.loaders.append(iter(loader))
        self.total_len = total

        print(f'{len(self.loaders)=}')


        print(f"✅ Combined dataset virtual size: {self.total_len} samples")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        """
        Random access: find which beton file this index falls into,
        load that sample, and return dict {'stems': ..., 'mix': ...}.
        """
        print(f'{idx=}')
        for start, end, beton_path, loader_idx in self.sample_offsets:
            print(f'{start=}, {end=}')
            if start <= idx < end:
                print('fwoop')
                with self.lock:
                    loader = self.loaders[loader_idx]
                    print(type(loader))
                    print('NEXT')
                    batch = next(loader)
                    print('DONE')
                stems = batch[0][0].cpu().numpy().astype('float32')
                mix = batch[1][0].cpu().numpy().astype('float32')
                # return {"stems": stems, "mix": mix}
                print('returning...')
                return (stems, mix)

                # local_idx = idx - start
                # loader = Loader(
                #     beton_path,
                #     batch_size=1,
                #     indices=[local_idx],
                #     num_workers=1,
                #     pipelines=self.pipelines,
                #     os_cache=False,
                #     drop_last=False,
                # )
                # for i, batch in enumerate(loader):
                #     # print(f'{i=}')
                #     # if i == local_idx:
                #     if True:
                #         stems = batch[0][0].cpu().numpy().astype('float32')
                #         mix = batch[1][0].cpu().numpy().astype('float32')
                #         # return {"stems": stems, "mix": mix}
                #         return (stems, mix)
        raise IndexError(f"Index {idx} out of range (max {self.total_len-1})")




def main():
    parser = argparse.ArgumentParser(description="Combine multiple latent .beton files into a single .beton file.")
    parser.add_argument("--in_dir", type=str, required=True,
                        help="Directory containing the input .beton files.")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path prefix for the combined output ('.beton' will be appended).")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for reading.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to load tensors on (cpu or cuda).")
    parser.add_argument("--num_stems", type=int, default=16,
                        help="Number of stems per sample.")
    parser.add_argument("--latent_size", type=int, default=64, 
                        help="Latent feature dimension size.")
    parser.add_argument("--latent_len", type=int, default=63,
                        help="Latent temporal length.")
    args = parser.parse_args()

    # Gather beton files
    beton_files = sorted([
        os.path.join(args.in_dir, f)
        for f in os.listdir(args.in_dir)
        if f.endswith(".beton")
    ])
    if not beton_files:
        raise FileNotFoundError(f"No .beton files found in {args.in_dir}")

    print(f"Found {len(beton_files)} beton files to combine.")

    # Define output schema
    schema = {
        "stems": NDArrayField(
            shape=(args.num_stems, args.latent_size, args.latent_len),
            dtype=np.dtype("float32")
        ),
        "mix": NDArrayField(
            shape=(args.latent_size, args.latent_len),
            dtype=np.dtype("float32")
        ),
    }

    print(f'expected size, stems: {(args.num_stems, args.latent_size, args.latent_len)}')
    print(f'expected size: mix:   {(args.latent_size, args.latent_len)}')


    # test next iter etc
    # pipelines = {
    #     "stems": [NDArrayDecoder(), ToTensor(), ToDevice(torch.device(args.device))],
    #     "mix": [NDArrayDecoder(), ToTensor(), ToDevice(torch.device(args.device))],
    # }
    # loader = Loader(
    #     beton_files[0],
    #     batch_size=1,
    #     num_workers=args.num_workers,
    #     pipelines=pipelines,
    #     os_cache=False,
    #     drop_last=False,
    # )
    # breakpoint()

    # Initialize writer
    # writer = DatasetWriter(f"{args.out_path}.beton", schema, num_workers=args.num_workers)
    writer = DatasetWriter(f"{args.out_path}.beton", schema, num_workers=1)

    lock = threading.Lock()
    # Build combined dataset
    # dataset = CombinedBetonDataset(
    #     beton_files=beton_files,
    #     num_workers=args.num_workers,
    #     batch_size=args.batch_size,
    #     device=args.device,
    #     lock=lock,
    # )

    dataset = BetonInMemory(
        beton_files=beton_files,
        num_workers=1,
        device=args.device,
    )

    # Stream-write combined dataset
    writer.from_indexed_dataset(dataset)
    print(f"✅ Combined dataset written to {args.out_path}.beton")

if __name__ == "__main__":
    main()
