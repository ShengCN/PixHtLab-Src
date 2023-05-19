import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Test_Dataset(Dataset):
    def __init__(self, opt, is_training=True):
        pass


    def __len__(self):
        return 100


    def __getitem__(self, idx):
        """ Return {'x': ..., 'y': ....}
        """
        x = torch.randn(3, 256, 256)
        y = torch.randn(3, 256, 256)

        return {'x': x, 'y': y}
