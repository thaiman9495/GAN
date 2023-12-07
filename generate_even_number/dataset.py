import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, lables):
        self.lables = lables

    def __len__(self):
        return len(self.lables)

    def __getitem__(self, item):
        return torch.FloatTensor(self.lables[item])
