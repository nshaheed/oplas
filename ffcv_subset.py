import argparse
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField, NDArrayField
from ffcv.loader import Loader
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder
from torch.utils.data import Dataset

import numpy as np


CHUNK_SIZE = 2**18  # Number of samples per chunk (262144)

class SubsetDataset(Dataset):
    def __init__(self, loader, num_samples):
        self.samples = []
        for i, batch in enumerate(loader):
            if len(self.samples) >= num_samples:
                break
            # unpack batch (assuming image + label)
            # images, labels = batch
            chunks, = batch
            # breakpoint()
            for chunk in chunks:
                self.samples.append(chunk)
            # for img, lbl in zip(images, labels):
            #     if len(self.samples) >= num_samples:
            #         break
            #     self.samples.append((img.numpy(), int(lbl)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx],)


def main(input_beton, output_beton, num_samples):
    PIPELINES = {
        "audio": [NDArrayDecoder()],
    }
    loader = Loader(
        input_beton,
        batch_size=1,
        num_workers=2,
        pipelines=PIPELINES,
        os_cache=False,  # can't cache bc it's too big to fit in memory
    )

    # Wrap into SubsetDataset
    dataset = SubsetDataset(loader, num_samples)

    # Define writer schema
    writer = DatasetWriter(
        output_beton,
        {"audio": NDArrayField(shape=(CHUNK_SIZE,), dtype=np.dtype("float32"))},
        # {
        #     'image': RGBImageField(write_mode='jpg', max_resolution=256, compress_probability=0.5),
        #     'label': IntField()
        # }
    )

    # Actually write the dataset
    writer.from_indexed_dataset(dataset)
    print(f"saved {len(dataset)} samples to {output_beton}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subset a .beton dataset")
    parser.add_argument("input_beton", type=str, help="Path to input .beton file")
    parser.add_argument("output_beton", type=str, help="Path to save subset .beton file")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples in subset")
    args = parser.parse_args()

    main(args.input_beton, args.output_beton, args.num_samples)
