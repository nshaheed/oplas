import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import librosa
import numpy as np
import umap

from music2latent import EncoderDecoder

def plot_pca_with_connections(a: torch.Tensor, b: torch.Tensor, batch_idx=0):
    """
    a, b: torch tensors of shape [batch_size, 64, t]
    batch_idx: which batch element to visualize
    """
    assert a.shape == b.shape, "Shapes of a and b must match"
    batch_size, dim, t = a.shape

    # Select one batch to visualize
    a_points = a[batch_idx].T  # shape [t, 64]
    b_points = b[batch_idx].T  # shape [t, 64]

    # Concatenate so PCA sees all points together
    all_points = torch.cat([a_points, b_points], dim=0).numpy()  # shape [2t, 64]

    # Apply PCA
    pca = PCA(n_components=64)
    reduced = pca.fit_transform(all_points)  # shape [2t, 2]

    print(pca.explained_variance_.tolist())

    # Split back into a and b
    a_reduced = reduced[:t]
    b_reduced = reduced[t:]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(a_reduced[:, 0], a_reduced[:, 1], c="blue", label="Orig")
    plt.scatter(b_reduced[:, 0], b_reduced[:, 1], c="red", label="offset 512 samp")

    # Draw lines between corresponding points
    for i in range(t):
        plt.plot([a_reduced[i, 0], b_reduced[i, 0]],
                 [a_reduced[i, 1], b_reduced[i, 1]], 
                 c="gray", linestyle="--", alpha=0.7)

    plt.legend()
    plt.title(f"PCA projection with connections (batch {batch_idx})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()


def plot_umap_with_connections(a: torch.Tensor, b: torch.Tensor, batch_idx=0):
    """
    a, b: torch tensors of shape [batch_size, 64, t]
    batch_idx: which batch element to visualize
    """
    assert a.shape == b.shape, "Shapes of a and b must match"
    batch_size, dim, t = a.shape

    # Select one batch to visualize
    a_points = a[batch_idx].T  # shape [t, 64]
    b_points = b[batch_idx].T  # shape [t, 64]

    # Concatenate so UMAP sees all points together
    all_points = torch.cat([a_points, b_points], dim=0).numpy()  # shape [2t, 64]

    # Apply UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(all_points)  # shape [2t, 2]

    # Split back into a and b
    a_reduced = reduced[:t]
    b_reduced = reduced[t:]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(a_reduced[:, 0], a_reduced[:, 1], c="blue", label="A")
    plt.scatter(b_reduced[:, 0], b_reduced[:, 1], c="red", label="B")

    # Draw lines between corresponding points
    for i in range(t):
        plt.plot([a_reduced[i, 0], b_reduced[i, 0]],
                 [a_reduced[i, 1], b_reduced[i, 1]], 
                 c="gray", linestyle="--", alpha=0.7)

    plt.legend()
    plt.title(f"UMAP projection with connections (batch {batch_idx})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    encdec = EncoderDecoder()

    
    file = "./kid_a.wav"
    wv, _ = librosa.load(file, sr=44100)
    wv = wv[:4*44100] # cut off
    # wv2 = np.concatenate((np.zeros(512), wv[:-512])) # pad to 512 sample
    wv2 = 0.5 * wv # check if gain is linear

    

    latent = encdec.encode(wv)
    latent2 = encdec.encode(wv2)

    # offset by 512 samples
    
    b, dim, t = 2, 64, 50
    # a = torch.randn(b, dim, t)
    # b = torch.randn(b, dim, t)
    plot_pca_with_connections(latent, latent2, batch_idx=0)
    # plot_umap_with_connections(latent, latent2, batch_idx=0)
