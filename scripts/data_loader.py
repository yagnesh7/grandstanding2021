import os.path as osp
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


# PyTorch Dataset
class AudioDataset(Dataset):
    def __init__(
        self, metadata, data_dir, y_name="gs_score", trunc_pad_len=2048, in_dim=5
    ):
        super().__init__()
        self.metadata = metadata

        # Faster than using a .loc on column names directly
        self.columns_dict = dict([(c, i) for i, c in enumerate(self.metadata.columns)])
        self.data_dir = data_dir
        self.y_name = y_name
        self.trunc_pad_len = trunc_pad_len
        self.in_dim = in_dim

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get row with .iloc
        row = self.metadata.iloc[idx]

        # Read in pre-computed numpy array
        file_name = row[self.columns_dict["file"]]
        line_name = row[self.columns_dict["line"]]
        npy_path = osp.join(self.data_dir, f"{file_name}_{line_name}.npy")
        data = np.load(npy_path)

        # Get y_true
        score = row[self.columns_dict[self.y_name]]

        # Pad/Truncate
        data_aug = np.zeros((self.trunc_pad_len, self.in_dim))
        data_aug[: min(data.shape[0], self.trunc_pad_len), :] = data[
            : self.trunc_pad_len
        ]

        item = {
            "x": torch.tensor(data_aug, dtype=torch.float),
            "y": torch.tensor([score], dtype=torch.float),
        }

        return item