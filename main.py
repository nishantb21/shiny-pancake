from sklearn.utils import resample
from model import ModelA
from helper import _get_sample

import torch

SAMPLE_RATE = 16000

m = ModelA()
inp, _ = _get_sample("org.flac", resample=SAMPLE_RATE)
inp = inp[:, :SAMPLE_RATE]
inp = torch.unsqueeze(inp, dim=0)

print(inp.shape)

output = m(inp)
output = torch.nn.AvgPool1d(2)(output)

print(output.shape)