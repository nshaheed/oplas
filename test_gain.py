from helper import get_gain
import torch

gain_effect = get_gain(device="cpu")


latent = torch.rand([64, 100])

result = gain_effect(latent, 0.01)

# should return same shape?
print(f"{result.shape=}, {latent.shape=}")

rms_norm = torch.nn.RMSNorm(latent.shape)
rms_norm = torch.nn.RMSNorm

for g in [
    5.0,
    4.0,
    3.0,
    2.0,
    1.0,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
    0.1,
    0.01,
    0.001,
    0.0,
]:
    result = gain_effect(latent, g)

    mse = torch.nn.functional.mse_loss(latent, result.permute(1, 0))

    print(f"Gain({g}): {mse}")
