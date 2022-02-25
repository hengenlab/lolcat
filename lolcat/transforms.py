import copy

import torch
from torch_geometric.data import Batch
import numpy as np


class Dropout:
    def __init__(self, dropout_p, apply_p=1.0, adaptive=False, randomized=False):
        self.dropout_p = dropout_p
        self.apply_p = apply_p
        self.adaptive = adaptive
        self.randomized = randomized

    def __call__(self, data):
        # make a copy from the data object
        data = copy.deepcopy(data)
        x = data.x
        if self.apply_p == 1. or torch.rand(1) < self.apply_p:
            if self.adaptive:
                dropout_p = 1 / (1 + x.sum(dim=1, keepdims=True))
                dropout_p = dropout_p/dropout_p.mean()
                dropout_p = dropout_p * self.dropout_p
                dropout_p = torch.clip(dropout_p, 0., self.dropout_p).squeeze()
            else:
                dropout_p = self.dropout_p

            if self.randomized:
                dropout_p *= torch.rand(1)

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
        x[:, self.std.squeeze() == 0.] = 0.
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


def mixup(y, batch, alpha=0.5):
    unique_class_ids = torch.unique(y)
    for unique_class_id in unique_class_ids:
        samples_from_same_class = torch.where(y == unique_class_id)[0]
        for i in range(samples_from_same_class.size(0) // 2):
            samples_to_mix = samples_from_same_class[i:i+2]

            sample_1_id, sample_2_id = samples_to_mix[0], samples_to_mix[1]

            sample_1_nodes = torch.where(batch == sample_1_id)[0]
            sample_2_nodes = torch.where(batch == sample_2_id)[0]

            lam = np.random.beta(alpha, alpha)
            num_trials_to_swap = int(min(sample_1_nodes.size(0),sample_2_nodes.size(0)) * lam)
            trials_1_to_swap = sample_1_nodes[torch.randperm(sample_1_nodes.size(0))[:num_trials_to_swap]]
            trials_2_to_swap = sample_2_nodes[torch.randperm(sample_2_nodes.size(0))[:num_trials_to_swap]]
            batch[trials_1_to_swap] = sample_2_id
            batch[trials_2_to_swap] = sample_1_id
    return batch
    
