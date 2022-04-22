import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

from helper import _get_sample

class CustomDataset(Dataset):
    def __init__(self):
        self.noise_path = "data/noiseData/"
        # self.clean_path = "data/cleanData/"
        self.noise_file_list = os.listdir(self.noise_path)
        # self.clean_file_list = os.listdir(self.clean_path)
        self.sample_rate = 16000
        self.clean_data = []
    
    def __len__(self):
        return len(self.noise_file_list)
    
    def __getitem__(self, idx):
        noisy_vector, _ = _get_sample(os.path.join(self.noise_path, self.noise_file_list[idx]), resample=self.sample_rate)
        # clean_vector = _get_sample(os.path.join(self.clean_path, self.clean_file_list[idx]), resample=self.sample_rate)
        # print(noisy_vector.shape)
        # print(self.sample_rate)
        noisy_vector = noisy_vector[:, :self.sample_rate]
        # clean_vector = clean_vector[:, :self.sample_rate]

        return noisy_vector #, clean_vector