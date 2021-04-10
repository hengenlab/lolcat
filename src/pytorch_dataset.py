from abc import ABC, abstractmethod

import inspect
import torch
from torch.utils.data import Dataset
import numpy as np
import sklearn

from src.covariance import compute_cov, compute_edge_dist
from src.isi import compute_isi_dist
from src.fr import compute_fr_dist



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

class FRDistributionDataset(PyTorchDataset):
    def __init__(self, master_dataset, scaler=sklearn.preprocessing.StandardScaler,
                 bins=200, **kwargs):
        super().__init__(master_dataset, **kwargs)

        assert 'transform' not in self.kwargs or self.kwargs['transform'] == 'firing_rate'
        self.kwargs['transform'] = 'firing_rate'

        self.scaler = scaler
        self.bins = bins

        if type(bins) == int:
            self.num_bins = bins
        else:
            self.num_bins = len(bins)-1

    def __getitem__(self, idx):
        trial_id = self.trials[idx]
        X, y, m = self.master_dataset.sample(**self.kwargs, trial_id=trial_id)
        fr_dist = compute_fr_dist(X.T, self.bins)
        normed_fr_dist = self.scaler.transform(np.vstack(fr_dist))
        fr_dist = list(normed_fr_dist)
        return fr_dist, y
    
class ISIDistributionDataset(PyTorchDataset):
    def __init__(self, master_dataset, scaler=sklearn.preprocessing.StandardScaler,
                 bins=200, min_isi=0, max_isi=0.4, **kwargs):
        super().__init__(master_dataset, **kwargs)
        
        assert 'transform' not in self.kwargs or self.kwargs['transform'] == 'interspike_interval'
        self.kwargs['transform'] = 'interspike_interval'
        
        self.bins = bins
        self.min_isi = min_isi
        self.max_isi = max_isi
        self.scaler = scaler
        
        if type(bins) == int:
            self.num_bins = bins
        else:
            self.num_bins = len(bins)-1
            
        
    def __getitem__(self, idx):
        trial_id = self.trials[idx]
        X, y, m = self.master_dataset.sample(**self.kwargs, trial_id=trial_id)
        isi_dist = compute_isi_dist(X, self.bins, self.min_isi, self.max_isi)
        normed_isi_dist = self.scaler.transform(np.vstack(isi_dist))
        isi_dist = list(normed_isi_dist)
        return isi_dist, y
    
class ISIFRDistributionDataset(PyTorchDataset):
    def __init__(self, master_dataset, isi_scaler=sklearn.preprocessing.StandardScaler, fr_scaler=sklearn.preprocessing.StandardScaler,
                 isi_bins=list(np.arange(0,0.402,0.002)), min_isi=0, max_isi=0.4, fr_bins=list(range(0,51,1)), **kwargs):
        super().__init__(master_dataset, **kwargs)
        
        assert 'transform' not in self.kwargs or self.kwargs['transform'] == 'interspike_interval'
        self.kwargs['transform'] = 'interspike_interval'
        
        self.isi_bins = self.isi_bins
        self.min_isi = min_isi
        self.max_isi = max_isi
        self.isi_scaler = isi_scaler
        
        self.fr_bins = fr_bins
        self.fr_scaler = fr_scaler
        
        if type(isi_bins) == int:
            self.isi_num_bins = isi_bins
        else:
            self.isi_num_bins = len(isi_bins)-1
        
        if type(fr_bins) == int:
            self.fr_num_bins = fr_bins
        else:
            self.fr_num_bins = len(fr_bins)-1
            
        
    def __getitem__(self, idx):
        trial_id = self.trials[idx]
        X, y, m = self.master_dataset.sample(**self.kwargs, trial_id=trial_id)
        isi_dist = compute_isi_dist(X, self.bins, self.min_isi, self.max_isi)
        normed_isi_dist = self.scaler.transform(np.vstack(isi_dist))
        isi_dist = list(normed_isi_dist)
        return isi_dist, y
