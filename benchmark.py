import argparse
import time
from pathlib import Path
from statistics import mean, stdev

import torch

parser = argparse.ArgumentParser(
    description="Run inference benchmarks on projector model."
)
parser.add_argument("checkpoint")
parser.add_argument(
    "-r",
    "--repetitions",
    default=100,
    type=int,
    help="number of times to repeat each benchmark",
)
parser.add_argument(
    "-b",
    "--batches",
    nargs="+",
    type=int,
    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    help="which batch sizes to run",
)
parser.add_argument(
    "-d", "--device", type=str, default="cpu", help="which device to run benchmark"
)
args = parser.parse_args()

device = torch.device(args.device)
projector = torch.load(
    args.checkpoint,
    map_location=device,
    weights_only=False,
)


# print(projector(input)[0].shape)

for batch in args.batches:
    repetitions = args.repetitions
    times = []
    for _ in range(repetitions):
        input = torch.randn(batch, 64).to(device)
        start = time.time()
        projector(input)
        end = time.time()
        times.append(end - start)

    avg_time = mean(times)
    avg_time_ms = avg_time * 1000  # convert to ms
    std_dev_ms = stdev(times) * 1000
    print(
        f"Batch size of {batch:4d} | mean: {avg_time_ms:02.3f}ms\t| std dev: {std_dev_ms:02.3f}ms"
    )
