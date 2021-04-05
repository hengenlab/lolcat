from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
import numpy as np
import sklearn

from src.covariance import compute_cov, compute_edge_dist


class PyTorchDataset(Dataset, ABC):
    r"""Abstract dataset for preaparing the dataset for training. The number of samples is equal to the number of
    trials. :obj:`kwargs` are passed along to master_dataset.sample. The number of elements in this dataset is equal
    to the number of trials in the split. So trial_random_seed and trial_id are not accepted.
    """
    def __init__(self, master_dataset, **kwargs):
        self.master_dataset = master_dataset
        self.kwargs = kwargs

        assert 'mode' in kwargs
        assert 'trial_random_seed' not in kwargs and 'trial_id' not in kwargs

        mode = self.kwargs['mode']
        self.trials = self.master_dataset.get_trials(mode)

    def __len__(self):
        return len(self.trials)

    @abstractmethod
    def __getitem__(self, idx):
        trial_id = self.trials[idx]
        X, y, m = self.master_dataset.sample(**self.kwargs, trial_id=trial_id)
        # perform transformations here
        pass

    @staticmethod
    def collate_fn(batch):
        data_list, label_list = [], []
        for _data, _label in batch:
            data_list.append(_data)
            label_list.append(_label)
        # stack
        data = np.vstack(data_list)
        label = np.concatenate(label_list)
        return torch.FloatTensor(data), torch.LongTensor(label)


class EdgeDistributionDataset(PyTorchDataset):
    def __init__(self, master_dataset, scaler=sklearn.preprocessing.StandardScaler,
                 edge_dist_num_bins=10, **kwargs):
        super().__init__(master_dataset, **kwargs)

        assert 'transform' not in self.kwargs or self.kwargs['transform'] == 'firing_rate'
        self.kwargs['transform'] = 'firing_rate'

        self.scaler = scaler
        self.num_bins = edge_dist_num_bins

    def __getitem__(self, idx):
        trial_id = self.trials[idx]
        X, y, m = self.master_dataset.sample(**self.kwargs, trial_id=trial_id)

        cov = compute_cov(X.T, scaler=self.scaler)
        edge_dist = compute_edge_dist(cov, num_bins=self.num_bins)  # matrix of shape (num_cells, num_bins)
        return edge_dist, y


class ISIDistributionDataset(PyTorchDataset):
    def __init__(self, master_dataset, scaler=sklearn.preprocessing.StandardScaler,
                 edge_dist_num_bins=10, **kwargs):
        super().__init__(master_dataset, **kwargs)

        assert 'transform' not in self.kwargs or self.kwargs['transform'] == 'interspike_interval'
        self.kwargs['transform'] = 'interspike_interval'

    def __getitem__(self, idx):
        trial_id = self.trials[idx]
        X, y, m = self.master_dataset.sample(**self.kwargs, trial_id=trial_id)
        raise NotImplementedError
