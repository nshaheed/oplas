from latent_effects import Gain
from helper import get_projector, get_gain, plot_audio, plot_audio2, slerp

import librosa
import torch
import torchaudio
import numpy as np

import torch
import torch.nn as nn

import random

from music2latent import EncoderDecoder

from tqdm import tqdm


# Create class
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

projector = get_projector(artifact_key="oauvlx7z", version="v142", device=device)
gain_effect = get_gain(device=device)

encdec = EncoderDecoder()

file = "./kid_a.wav"
wv, _ = librosa.load(file, sr=44100)
wv = 1.0 * wv[590150 : 44100 * 14]

kick = encdec.encode(wv).squeeze().permute(1, 0)

# going to test linearity of sample-level offsets
xs = []
ys = []
for i in tqdm(range(5632)):
    # for i in tqdm(range(1000)):
    offset = np.concatenate((np.zeros(i), wv))
    silence = encdec.encode(offset).squeeze()

    x = torch.concat((torch.tensor([i]), kick[0]))
    y = silence[:, 0]

    xs.append(x)
    ys.append(y)

    # breakpoint()
    # data.append((silence[:,0], i))

# breakpoint()
x_train = torch.stack(xs)
y_train = torch.stack(ys)


model = LinearRegressionModel(65, 64)

criterion = nn.MSELoss()

learning_rate = 0.000005

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 1000000

for epoch in tqdm(range(epochs)):
    epoch += 1
    # Convert numpy array to torch Variable
    # inputs = torch.from_numpy(x_train).requires_grad_()
    # labels = torch.from_numpy(y_train)

    inputs = x_train
    labels = y_train

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward to get output
    outputs = model(inputs)

    # Calculate Loss
    loss = criterion(outputs, labels)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    print("epoch {}, loss {}".format(epoch, loss.item()))


# result = []
# # gradually interpolate between the two
# for i in np.linspace(0,1,10):
#     # interp = (1-i) * kick + i * silence
#     interp = slerp(kick, silence, i)
#     audio = encdec.decode(interp.permute(1,0))

#     torchaudio.save(f"offset_slerp-{i}.wav", audio, 44100)


# latent = torch.concat(result[:1], 0).permute(1,0)

# audio = encdec.decode(kick[:1].permute(1,0))

# torchaudio.save(f"single_latent.wav", audio, 44100)
