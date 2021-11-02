import copy

import torch
from torch_geometric.data import Batch


class Dropout:
    def __init__(self, dropout_p):
        self.dropout_p = dropout_p

    def __call__(self, data):
        # make a copy from the data object
        data = copy.deepcopy(data)
        x = data.x

        dropout_p = 1 / (1 + x.sum(dim=1, keepdims=True))
        dropout_p = dropout_p/dropout_p.mean()
        dropout_p = dropout_p * self.dropout_p
        dropout_p = torch.clip(dropout_p, 0., 0.7).squeeze()

        # dropout entire trials
        dropout_mask = torch.empty((x.size(0),), dtype=torch.float32, device=x.device).uniform_(0, 1) > dropout_p
        data.x = x[dropout_mask]
        return data


class Normalize:
    def __init__(self, mean, std, copy=True):
        self.mean = mean
        self.std = std
        self.copy=copy

    def __call__(self, data):
        if self.copy:
            data = copy.deepcopy(data)
        x = data.x
        x = (x-self.mean.to(x.device)) / self.std.to(x.device)
        data.x = x
        return data

    def unnormalize_x(self, x):
        return (x * self.std.to(x.device)) + self.mean.to(x.device)


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


def compute_mean_std(dataset):
    data = Batch.from_data_list(dataset)
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False, keepdim=True)
    return mean, std
