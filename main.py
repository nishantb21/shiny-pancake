from sklearn.utils import resample
from model import ModelA
from helper import _get_sample

import torch

# CONSTANTS
SAMPLE_RATE = 16000
BATCH_SIZE = 1
EPOCHS = 1

# Dataset object
# TODO: Add parameters
# TODO: Uncomment the clean speech

# TODO: Convert Dataset object to Dataloader

# TODO: Define the model (!leave this part)

# TODO: Define your loss function

# TODO: Set up optimizer
# torch.optim.SGD(model.paramters())

# TODO: Training loop

# for batch in dataloader_object:
#     noisy_batch, clean_batch = batch
    # optimizer.zero_grad()

    # TODO: Feed noisy into model
    # TODO: Compute loss between the output and clean_batch
    # TODO: Compute gradients: loss.backward()
    # TODO: Apply them to the network: optimizer.step()