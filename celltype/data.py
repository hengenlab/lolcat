import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import os
from src import Dataset as CellTypeDataset
from torch.utils.data.sampler import SubsetRandomSampler


from abc import ABC, abstractmethod


class CellSets(Dataset, ABC):
    def __init__(self, name, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__()
        self.name = name
        self.root = root
        self.random_seed = random_seed
        self.num_bins = num_bins

        assert split in ['train', 'val', 'test']
        self.split = split

        self.transform = transform

        # check if already processed
        already_processed, filename = self._look_for_processed_file()

        # if not processed or force_process
        if not (already_processed) or force_process:
            print('Processing data.')
            # process and save data
            self.process()

        self.cell_data, self.class_names, self.target = self.load()

    def __getitem__(self, item):
        data = self.cell_data[item]
        if self.transform is not None:
           data = self.transform(data)
        return data

    def __len__(self):
        return len(self.cell_data)

    def processed_filename(self, split):
        return os.path.join(self.root, '{}-data-seed{}-num_bins-{}-{}split.pt'.format(self.name, self.random_seed, self.num_bins, split))

    def _look_for_processed_file(self):
        filename = self.processed_filename(self.split)
        return os.path.exists(filename), filename

    def load(self):
        filename = self.processed_filename(self.split)
        processed = torch.load(filename)
        return processed['data'], processed['class_names'], processed['target']

    def process(self):
        data, class_names = self.get_data()

        # compute histograms
        data['train'][0] = compute_log_isi_distribution(data['train'][0], num_bins=self.num_bins)
        data['val'][0] = compute_log_isi_distribution(data['val'][0], num_bins=self.num_bins)
        data['test'][0] = compute_log_isi_distribution(data['test'][0], num_bins=self.num_bins)

        # convert to graphs
        train_data = self.convert_to_graph(*data['train'])
        val_data = self.convert_to_graph(*data['val'])
        test_data = self.convert_to_graph(*data['test'])

        # save
        torch.save({'data': train_data, 'class_names': class_names, 'target': torch.LongTensor(data['train'][1])}, self.processed_filename('train'))
        torch.save({'data': val_data, 'class_names': class_names, 'target': torch.LongTensor(data['val'][1])}, self.processed_filename('val'))
        torch.save({'data': test_data, 'class_names': class_names, 'target': torch.LongTensor(data['test'][1])}, self.processed_filename('test'))

    def convert_to_graph(self, X, cell_type, cell_index, **kwargs):
        # group all the data from each cell in the same set
        cell_data = []
        for cell_id in range(cell_type.shape[0]):
            cell_mask = cell_index == cell_id
            x = X[cell_mask]
            x = torch.FloatTensor(x)
            y = torch.tensor(cell_type[cell_id]).long()
            other_labels = {}
            for key in kwargs:
                other_labels[key] = torch.tensor(kwargs[key][cell_id])
            data = Data(x=x, y=y, **other_labels)
            cell_data.append(data)
        return cell_data

    def to(self, device):
        [data.to(device) for data in self.cell_data]
        return self

    @abstractmethod
    def get_data(self, test_size=0.2, val_size=0.2):
        pass

    @abstractmethod
    def get_sampler(self):
        pass


def compute_log_isi_distribution(X, num_bins=128, a_min=-2.5, a_max=0.5):
    X_isi = np.zeros((X.shape[0], num_bins))
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


### V1 Dataset
class V1CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('v1', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='v1', labels_col='pop_name', force_process=False)

        # each sample must have at least 30 spikes
        dataset.drop_dead_cells(cutoff=30)

        # cell classes identified by Louis as not being too quiet
        keepers = ['e5Rbp4', 'e23Cux2', 'i6Pvalb', 'e4Scnn1a', 'i23Pvalb', 'i23Htr3a',
                   'e4Rorb', 'e4other', 'i5Pvalb', 'i4Pvalb', 'i23Sst', 'i4Sst', 'e4Nr5a1',
                   'i1Htr3a', 'e5noRbp4', 'i6Sst', 'e6Ntsr1']
        dataset.drop_other_classes(classes_to_keep=keepers)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, keepers

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(9 - torch.log(class_sample_count))))
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3,4, 5]
        return SubsetRandomSampler(indices)


# todo adapt to task/ dataset
class NeuropixelsBrainSturcture5CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('neuropixels_brain_structure_5', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='v1', labels_col='pop_name', force_process=False)

        # each sample must have at least 30 spikes
        dataset.drop_dead_cells(cutoff=30)

        # cell classes identified by Louis as not being too quiet
        class_names = ['e5Rbp4', 'e23Cux2', 'i6Pvalb', 'e4Scnn1a', 'i23Pvalb', 'i23Htr3a',
                       'e4Rorb', 'e4other', 'i5Pvalb', 'i4Pvalb', 'i23Sst', 'i4Sst', 'e4Nr5a1',
                       'i1Htr3a', 'e5noRbp4', 'i6Sst', 'e6Ntsr1']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        return None
