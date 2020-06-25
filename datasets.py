import torch
import numpy as np
from torch.utils.data import Dataset

class BallDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.obj_data=np.load('obj_data.npy')
        self.adj_data=np.load('adj_data.npy')
        self.label_data=np.load('label_data.npy')
        self.size=self.obj_data.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.obj_data[idx], self.adj_data[idx], self.label_data[idx]