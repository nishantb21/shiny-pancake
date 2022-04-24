from sklearn.utils import resample
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import SGD
import torch

from model import ModelA, ModelB
from helper import _get_sample
from dataset import CustomDataset

# CONSTANTS
SAMPLE_RATE = 16000
BATCH_SIZE = 1
EPOCHS = 1
noisy_path = "assets/noisySpeech"
clean_path = "assets/cleanSpeech"

dataset = CustomDataset(noisy_path, clean_path)
dataloader = DataLoader(dataset,batch_size= BATCH_SIZE, shuffle=True)

model = ModelB()
criterion = MSELoss()
optim = SGD(model.parameters())

for batch in dataloader:
    noisy_batch, clean_batch = batch
    optim.zero_grad()

    output = model(noisy_batch)
    loss = criterion(clean_batch, output)
    loss.backward()
    print("Loss: ", loss.item())
    optim.step()