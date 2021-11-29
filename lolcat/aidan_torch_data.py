from abc import ABC, abstractmethod
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch_geometric.data import Data

from lolcat.data import V1Dataset, CalciumDataset


class InMemoryDataset(Dataset, ABC):
    processed_dir = 'processed_pt/'

    def __init__(self, name, root, split, target, *, random_seed=123, num_bins=128, transform=None, force_process=False, lite=True):
        super().__init__()
        self.name = name
        self.root = root
        self.lite = lite

        self.random_seed = random_seed
        self.num_bins = num_bins

        assert split in ['train', 'val', 'test']
        self.split = split

        self.transform = transform

        # check if already processed
        already_processed, filename = self._look_for_processed_file()

        # if not processed or force_process
        if not (already_processed) or force_process:
            print('Transforming data to tensor format.')
            # process and save data
            self.process()

        self.data_list = self.load()
        self.set_target(target)

    def __getitem__(self, item):
        if self.lite:
            data = Data(x=self.data_list[item].x)
        else:
            data = self.data_list[item]
        data.y = self.target[item]
        if self.transform is not None:
           data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data_list)

    def processed_filename(self, split):
        return os.path.join(self.root, self.processed_dir,
                            '{}-data-seed:{}-num_bins:{}-{}_split.pt'.format(
                                self.name, self.random_seed, self.num_bins, split))

    def _look_for_processed_file(self):
        filename = self.processed_filename(self.split)
        return os.path.exists(filename), filename

    def load(self):
        filename = self.processed_filename(self.split)
        processed = torch.load(filename)
        return processed['data_list']

    def to_graph(self, data):
        for key in data.keys():
            try:
                # only convert structured np arrays to tensors
                data[key] = torch.tensor(data[key])
            except:
                # do nothing
                pass

        return Data(**data)

    def to(self, device):
        # move features x and y only to gpu
        for data in self.data_list:
            data.x.to(device)
        self.target.to(device)
        return self

    def process(self):
        dataset = self.prepare_dataset()

        for split in ['train', 'val', 'test']:
            data_list = dataset.get_data(split)

            graph_list = []
            for data in data_list:
                data = self.compute_feats(data)
                data = self.to_graph(data)
                if self.filter_data(data):
                    graph_list.append(data)

            torch.save({'data_list': graph_list}, self.processed_filename(split))

    @abstractmethod
    def filter_data(self, data):
        ...

    @abstractmethod
    def compute_feats(self, data):
        ...
        #data['x'] = compute_log_isi_distribution(data['spikes'], num_bins=self.num_bins)
        # compute_isi_distribution(data['train']['X'], num_bins=self.num_bins, add_origin=True)
        # compute_isi_distribution(data['train']['X_block'], num_bins=180, a_min=0., a_max=6.0)

    @abstractmethod
    def prepare_dataset(self, test_size=0.2, val_size=0.2):
        ...

    def set_target(self, label):
        assert label in self.data_list[0].keys
        filename = self.processed_filename(self.split)
        filename = '{}_{}.pt'.format(os.path.splitext(filename)[0], label)
        if not os.path.exists(filename):
            target = np.array([d[label] for d in self.data_list])
            class_names, target = np.unique(target, return_inverse=True)
            target = torch.LongTensor(target)
            torch.save({'target': target, 'class_names': class_names}, filename)
        processed = torch.load(filename)
        self.target, self.class_names = processed['target'], list(processed['class_names'])

    @property
    def increase_factor(self):
        # by default, balanced sampler
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        return 1 / class_sample_count

    def get_sampler(self):
        # class-wise resampling
        indices = []
        for cell_type, factor in enumerate(self.increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)
        return SubsetRandomSampler(indices)


######
# V1 #
######
class V1DGTorchDataset(InMemoryDataset):
    def __init__(self, root, split, target, k, *, random_seed=123, num_bins=128, transform=None, force_process=False, lite=True):
        self.k = k
        name = 'v1_{}'.format(self.k)
        super().__init__(name, root, split, target, random_seed=random_seed, num_bins=num_bins, transform=transform,
                         force_process=force_process, lite=lite)

    def prepare_dataset(self, test_size=0.2, val_size=0.2):
        dataset = V1Dataset(self.root)

        # each sample must have at least 30 spikes
        dataset.drop_dead_cells(cutoff=30)

        if self.k == '13':
            dataset.filter_cells('13celltypes', keep=['e23', 'e6', 'i5Htr3a', 'i6Pvalb', 'i5Pvalb', 'i6Sst', 'i4Htr3a',
                                                      'i23Htr3a', 'e4', 'i1Htr3a', 'i4Sst', 'e5', 'i23Sst', 'i4Pvalb',
                                                      'i5Sst', 'i6Htr3a', 'i23Pvalb'])
        else:
            raise NotImplementedError

        # todo better startification before filtering.
        dataset.train_val_test_split(test_size=test_size, val_size=val_size, random_seed=self.random_seed,
                                     stratify_by='13celltypes')
        return dataset

    def compute_feats(self, data):
        data['x'] = compute_log_isi_distribution(data['spikes'], num_bins=self.num_bins, a_min=-2.5, a_max=0.5)
        # data['x_global'] = compute_log_isi_distribution(data['spike_blocks'], num_bins=self.num_bins, a_min=-2.5, a_max=0.5)
        return data

    def filter_data(self, data):
        return True

    @property
    def increase_factor(self):
        if self.k == '13':
            increase_factor = torch.FloatTensor([1., 1., 2., 1., 3., 3., 3., 2., 3., 3., 3., 2., 1.])
        else:
            raise NotImplementedError
        return increase_factor



###########
# CALCIUM #
###########
class CalciumDGTorchDataset(InMemoryDataset):
    stimulus = 'drifting_gratings'

    def __init__(self, root, split, k, *, random_seed=123, num_bins=90, transform=None, force_process=False, lite=True):
        self.k = k
        target = {'4': '4newcelltypes'}[self.k]
        name = 'calcium_{}_{}'.format(self.stimulus, self.k)
        super().__init__(name, root, split, target, random_seed=random_seed, num_bins=num_bins, transform=transform,
                         force_process=force_process, lite=lite)

    def prepare_dataset(self, test_size=0.2, val_size=0.2):
        dataset = CalciumDataset(self.root, self.stimulus)
        if self.k == '2':
            dataset.filter_cells('class', keep=['e', 'i'])
        elif self.k == '4':
            dataset.filter_cells('4newcelltypes', keep=['e', 'Vip', 'Sst', 'Pvalb'])
        elif self.k == '5':
            dataset.filter_cells('5newcelltypes', keep=['Cux2', 'Sst', 'Vip', 'Pvalb', 'Rorb'])
        elif self.k == '6':
            dataset.filter_cells('6newcelltypes', keep=['Cux2', 'Sst', 'Ntsr1', 'Vip', 'Pvalb', 'Rorb'])
        elif self.k == '7':
            dataset.filter_cells('7newcelltypes', keep=['e4', 'Sst', 'Vip', 'Pvalb', 'e23', 'e6', 'e5'])
        elif self.k == '13':
            dataset.filter_cells('subclass_full', keep=['Rbp4', 'Slc17a7', 'Cux2', 'Fezf2', 'Ntsr1', 'Emx1', 'Sst',
                                                        'Tlx3', 'Scnn1a', 'Rorb', 'Nr5a1', 'Pvalb', 'Vip'])
        else:
            raise NotImplementedError

        # todo better startification before filtering.
        
        ### LOOK HERE ####
        if isinstance(self.randomseed,str):
            if self.stimulus == 'drifting_gratings':
                dataset.load_train_val_test_split('./cellsplits/calcium_drifting_gratings_4celltypes_cell_split_{}.csv'.format(str(split_seed)))
            elif self.stimulus == 'natural_movies':
                dataset.load_train_val_test_split('./cellsplits/calcium_natural_movie_three_4celltypes_cell_split_{}.csv'.format(str(split_seed)))
                
        else:
            dataset.train_val_test_split(test_size=test_size, val_size=val_size, random_seed=self.random_seed, stratify_by='subclass_full')
        return dataset

    def compute_feats(self, data):
        data['x'] = compute_isi_distribution(data['spikes'], num_bins=self.num_bins, a_min=0., a_max=3.0, add_origin=True)
        data['x_global'] = compute_isi_distribution(data['spike_blocks'], num_bins=180, a_min=0., a_max=6.0)
        return data

    def filter_data(self, data, thresh=5.):
        return (data.x.sum(dim=1) >= thresh).sum() > 0

    @property
    def increase_factor(self):
        if self.k == '4':
            increase_factor = torch.FloatTensor([50., 50, 20, 1.])
        elif self.k == '7':
            #todo adjust
            increase_factor = torch.FloatTensor([1., 1., 1., 1., 1., 1., 1.])
        elif self.k == '6':
            increase_factor = torch.FloatTensor([1., 1., 1., 1., 1., 1.])
        else:
            raise NotImplementedError
        return increase_factor


class CalciumNMTorchDataset(CalciumDGTorchDataset):
    stimulus = 'naturalistic_movies'



#########
# Utils #
#########
def compute_log_isi_distribution(X, num_bins=128, a_min=-2.5, a_max=0.5):
    X_isi = np.zeros((len(X), num_bins))
    bins = np.linspace(a_min, a_max, num_bins + 1)
    for i, x in enumerate(X):
        # compute isi
        x = np.diff(x)
        # transform to log scale
        x = np.log10(x)
        # clip
        x = np.clip(x, a_min=a_min, a_max=a_max)
        X_isi[i] = np.histogram(x, bins)[0].astype(int)
    return X_isi


def compute_isi_distribution(X, num_bins=128, a_min=0., a_max=3.0, add_origin=False):
    X_isi = np.zeros((len(X), num_bins))
    bins = np.linspace(a_min, a_max, num_bins + 1)
    for i, x in enumerate(X):
        if add_origin:
            x = np.concatenate([[1e-9], x])
        # compute isi
        x = np.diff(x)
        # clip
        x = np.clip(x, a_min=a_min, a_max=a_max)
        X_isi[i] = np.histogram(x, bins)[0].astype(int)
    return X_isi


def compute_fr_distribution(X, num_bins=90, a_min=0., a_max=3.0):
    X_fr = np.zeros((X.shape[0], num_bins))
    bins = np.linspace(a_min, a_max, num_bins + 1)
    raise NotImplementedError
    for i, x in enumerate(X):
        bins = np.linspace(a_min, a_max, num_bins + 1)
        # compute isi
        x = np.diff(x)
        # clip
        x = np.clip(x, a_min=a_min, a_max=a_max)
        X_isi[i] = np.histogram(x, bins)[0].astype(int)
    return X_isi