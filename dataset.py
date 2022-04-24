import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

from helper import _get_sample

class CustomDataset(Dataset):
    def __init__(self, noisy_path, clean_path):
        self.noisy_path = noisy_path
        self.clean_path = clean_path
        self.noisy_file_list = os.listdir(self.noisy_path)
        # self.clean_file_list = os.listdir(self.clean_path)
        self.sample_rate = 16000
        self.clean_data = []
    
    def __len__(self):
        return len(self.noisy_file_list)
    
    def __getitem__(self, idx):
        noisy_vector, _ = _get_sample(os.path.join(self.noisy_path, self.noisy_file_list[idx]), resample=self.sample_rate)
        clean_file = self.noisy_file_list[idx].split('_')[0] + '.flac'
        clean_vector = _get_sample(os.path.join(self.clean_path, clean_file), resample=self.sample_rate)
        # print(noisy_vector.shape)
        # print(self.sample_rate)
        noisy_vector = noisy_vector[:, :self.sample_rate]
        clean_vector = clean_vector[:, :self.sample_rate]

        return noisy_vector, clean_vector