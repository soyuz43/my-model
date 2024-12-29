# utils/hdf5_dataset.py

import torch
from torch.utils.data import Dataset
import h5py

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, sequence_length):
        self.hdf5_path = hdf5_path
        self.sequence_length = sequence_length
        self.file = h5py.File(self.hdf5_path, 'r')
        self.tokens = self.file['tokens']
        self.length = len(self.tokens) - self.sequence_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx
        end = start + self.sequence_length + 1  # +1 for the next token
        token_seq = self.tokens[start:end]
        input_seq = torch.tensor(token_seq[:-1], dtype=torch.long)
        target = torch.tensor(token_seq[-1], dtype=torch.long)  # Only the next token
        return input_seq, target
