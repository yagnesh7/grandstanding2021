import os.path as osp
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


# PyTorch Dataset
class AudioDatasetWithStats(Dataset):
    def __init__(
        self,
        metadata,
        data_dir,
        y_name="gs_score",
        trunc_pad_len=2048,
        in_dim=35,
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

        means_stds = [
            c for c in self.columns_dict.keys() if ("mean" in c) or ("std" in c)
        ]

        # Read in pre-computed numpy array
        file_name = row[self.columns_dict["file"]]
        line_name = row[self.columns_dict["line"]]
        npy_path = osp.join(self.data_dir, f"{file_name}_{line_name}.npy")
        data = np.load(npy_path)

        # Get y_true
        score = row[self.columns_dict[self.y_name]]

        # Get averages and standard deviations of the features before padding.
        summary_arr = row[[self.columns_dict[c] for c in means_stds]].values
        summary_arr_tiled = np.tile(summary_arr, (data.shape[0], 1))

        data = np.concatenate([data, summary_arr_tiled], axis=1)

        # Pad/Truncate
        data_aug = np.zeros((self.trunc_pad_len, self.in_dim))
        data_aug[: min(data.shape[0], self.trunc_pad_len), :] = data[
            : self.trunc_pad_len
        ]
        item = {
            "x": torch.tensor(data_aug, dtype=torch.float),
            "y": torch.tensor([score], dtype=torch.float),
        }

        return (item["x"], item["y"])


class AudioDataset(Dataset):
    def __init__(
        self,
        metadata,
        data_dir,
        y_name="gs_score",
        trunc_pad_len=2048,
        in_dim=35,
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

        return (item["x"], item["y"])


class TimeseriesDataset(Dataset):
    """
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len - 1)

    def __getitem__(self, index):
        return (self.X[index : index + self.seq_len], self.y[index + self.seq_len - 1])


class NewAudioDataset(Dataset):
    def __init__(
        self,
        metadata,
        data_maxes=np.load("../outputs/data_maxes.npy"),
        data_directory="../outputs/npy2",
        num_features: int = 5,
        seq_len: int = 2048,
        y_col="gs_score",
    ):
        self.metadata = metadata
        self.columns_dict = dict([(c, i) for i, c in enumerate(self.metadata.columns)])
        self.data_maxes = data_maxes
        self.data_directory = data_directory
        self.num_features = num_features
        self.seq_len = seq_len
        self.y_col = "gs_score"

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        # means_stds = [c for c in self.columns_dict.keys() if ("mean" in c) or ("std" in c)]
        file_name = row[self.columns_dict["file"]]
        line_name = row[self.columns_dict["line"]]
        npy_path = osp.join(self.data_directory, f"{file_name}_{line_name}.npy")
        data = np.load(npy_path)
        data = data / self.data_maxes

        # Get y_true
        score = row[self.columns_dict[self.y_col]]
        #         score = score.reshape(-1,1)

        data_aug = np.zeros((self.seq_len, self.num_features))

        data_aug[: min(data.shape[0], self.seq_len), :] = data[: self.seq_len]

        item = {
            "x": torch.tensor(data_aug, dtype=torch.float),
            "y": torch.tensor(score, dtype=torch.float),
        }

        return (item["x"], item["y"])


class HUBERTDataset(torch.utils.data.Dataset):
    def __init__(self, mp3_clips, summary_data, pt_path):
        self.mp3_clips = mp3_clips
        self.summary_data = summary_data
        self.pt_path = pt_path

    def __getitem__(self, idx):
        file, line = self.mp3_clips[idx]
        input_values = torch.load(f"{self.pt_path}{file}-{line}.pt")
        input_values.requires_grad = False

        if input_values.size()[0] != 1:
            input_values = input_values.mean(dim=0)
        else:
            input_values = input_values.squeeze(0)
        labels = self.summary_data.loc[
            (self.summary_data["file"] == file) & (self.summary_data["line"] == line),
            "gs_score",
        ].values[0]
        labels = torch.tensor(labels).unsqueeze(0)
        return {"input_values": input_values.float(), "labels": labels.float()}

    def __len__(self):
        return len(self.mp3_clips)
