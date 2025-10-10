from oplas.data import MTGJamendoStream, MTGJamendoStreamSingle

import time
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

# assuming MTGJamendoStreamSingle is defined elsewhere
data_dir = "/scratch/users/nshaheed/mtg-jamendo-wav/"
dataset = MTGJamendoStreamSingle(data_dir=data_dir, load_frac=0.5)

batch_size = 64

num_workers_list = [128, 64, 48, 32, 24, 16, 8]
prefetch_factors = [128, 64, 32, 16, 8, 4, 2]
persistent_options = [False, True]

csv_file = "dataloader_benchmark_results.csv"

# initialize CSV with header
with open(csv_file, mode="w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["num_workers", "prefetch_factor", "persistent_workers", "time"]
    )
    writer.writeheader()

for num_workers in num_workers_list:
    for prefetch in prefetch_factors:
        for persistent in persistent_options:
            print(
                f"\nConfig: num_workers={num_workers}, prefetch_factor={prefetch}, persistent_workers={persistent}"
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                shuffle=True,
                persistent_workers=persistent,
                prefetch_factor=prefetch
                if num_workers > 0
                else None,  # only valid if workers > 0
            )

            start_time = time.time()
            for batch in tqdm(dataloader, smoothing=0, desc="Iterating"):
                # just consume the batch, no processing
                pass
            total_time = time.time() - start_time

            print(f"Total time: {total_time:.2f}s")

            # append result to CSV
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "num_workers",
                        "prefetch_factor",
                        "persistent_workers",
                        "time",
                    ],
                )
                writer.writerow(
                    {
                        "num_workers": num_workers,
                        "prefetch_factor": prefetch,
                        "persistent_workers": persistent,
                        "time": total_time,
                    }
                )

print("\nBenchmark complete. Results saved to", csv_file)
