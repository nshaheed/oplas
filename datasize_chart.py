import matplotlib.pyplot as plt
import glob

# Adjust this pattern to match your files
# e.g. "data/*.txt" or "*.csv"
file_pattern = "multivoice_benchmarks/*.csv"

# 1/256, 1/128, etc
data_frac =[ 256, 128, 64, 32, 16, 8, 4, 2, 1]
files = [ "s52zbb9h", "0t0ifbvt", "d8gkgiu9", "rrx9gt7v", "otjkc28v", "ek6cahrw", "t2ib5c80", "qahnouow", "xz9pve6h" ]


data_sets = []

# Loop through all matching files
for model, frac in zip(files, data_frac):
    filename = f"multivoice_benchmarks/{model}_latest.csv"
    x_vals = []
    y_vals = []

    with open(filename) as f:
        next(f)
        for line in f:
            if line.strip():  # skip empty lines
                x, y = line.strip().split(",")
                x_vals.append(float(x))
                y_vals.append(float(y))

    sorted_pairs = sorted(zip(x_vals, y_vals))
    x_sorted, y_sorted = zip(*sorted_pairs)
    data_sets.append((frac, x_sorted, y_sorted))
    # plt.plot(x_sorted, y_sorted, marker='', linestyle='-', label=voices)

# -------------------------------
# Create one figure with two subplots
# -------------------------------
plt.style.use('tableau-colorblind10')
# fig, axes = plt.subplots(1, 1, figsize=(12, 12), sharey=False)
fig, axes = plt.subplots(1, 1, sharey=False)

# --- top subplots: zoomed (x ≤ 16)
for frac, x, y in data_sets:
    zoom_x = [xv for xv in x if xv <= 16]
    zoom_y = [yv for xv, yv in zip(x, y) if xv <= 16]
    label=f"1/{frac} of dataset"
    if frac == 1:
        label = "full dataset"
    axes.plot(x, y, marker='', linestyle='-', label=f"1/{frac} of dataset")
axes.set_xlabel("Number of Voices")
axes.set_ylabel("Reconstruction Loss")
axes.set_title("Mixing Loss")
# axes.legend(title="Fraction of dataset")
axes.legend()
axes.grid(True)

# for voices, x, y in data_sets:
#     zoom_x = [xv for xv in x if xv <= 16]
#     zoom_y = [yv for xv, yv in zip(x, y) if xv <= 16]
#     axes[0][1].plot(zoom_x, zoom_y, marker='', linestyle='-', label=f"{voices}-voice model")
# axes[0][1].set_xlabel("Number of Voices")
# axes[0][1].set_ylabel("Reconstruction Loss")
# axes[0][1].set_ylim([0, 0.25])
# axes[0][1].set_title("Mixing Loss Up To 16 Voices")
# axes[0][1].legend(title="Max number of voices models trained on")
# axes[0][1].grid(True)

# # --- Right subplot: full range
# for voices, x, y in data_sets:
#     axes[1][0].plot(x, y, marker='', linestyle='-', label=f"{voices}-voice model")
# axes[1][0].set_xlabel("Number of Voices")
# axes[1][0].set_ylabel("Reconstruction Loss")
# axes[1][0].set_title("Mixing Loss Up To 128 Voices")
# axes[1][0].legend(title="Max number of voices models trained on")
# axes[1][0].grid(True)

# # --- Right subplot: full range
# for voices, x, y in data_sets:
#     axes[1][1].plot(x, y, marker='', linestyle='-', label=f"{voices}-voice model")
# axes[1][1].set_xlabel("Number of Voices")
# axes[1][1].set_ylabel("Reconstruction Loss")
# axes[1][1].set_ylim([0, 2])
# axes[1][1].set_title("Mixing Loss Up To 128 Voices")
# axes[1][1].legend(title="Max number of voices models trained on")
# axes[1][1].grid(True)

# plt.tight_layout()

# Save the combined figure
plt.savefig("datasize_benchmark.png", dpi=300)
