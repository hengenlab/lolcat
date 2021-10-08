import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import os
from src import Dataset as CellTypeDataset
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

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
        data['train']['X'] = compute_log_isi_distribution(data['train']['X'], num_bins=self.num_bins)
        data['val']['X'] = compute_log_isi_distribution(data['val']['X'], num_bins=self.num_bins)
        data['test']['X'] = compute_log_isi_distribution(data['test']['X'], num_bins=self.num_bins)

        # convert to graphs
        train_data = self.convert_to_graph(**data['train'])
        val_data = self.convert_to_graph(**data['val'])
        test_data = self.convert_to_graph(**data['test'])

        # save
        torch.save({'data': train_data, 'class_names': class_names, 'target': torch.LongTensor(data['train']['cell_type'])}, self.processed_filename('train'))
        torch.save({'data': val_data, 'class_names': class_names, 'target': torch.LongTensor(data['val']['cell_type'])}, self.processed_filename('val'))
        torch.save({'data': test_data, 'class_names': class_names, 'target': torch.LongTensor(data['test']['cell_type'])}, self.processed_filename('test'))

    def convert_to_graph(self, X, cell_type, cell_index, trial_metadata, **kwargs):
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
            for key in trial_metadata:
                other_labels[key] = torch.tensor(trial_metadata[key])

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
        if isinstance(x,int):
            x = np.array([])
        # compute isi
        x = np.diff(x)
        # transform to log scale
        x = np.log10(x)
        # clip
        x = np.clip(x, a_min=a_min, a_max=a_max)
        X_isi[i] = np.histogram(x, bins)[0].astype(int)
    return X_isi

### V1 Dataset
class V1Types17CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('v1_celltypes_17', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2, label_col='17celltypes'):
        dataset = CellTypeDataset(self.root, data_source='v1', labels_col='17celltypes', force_process=False)

        # each sample must have at least 30 spikes
        dataset.drop_dead_cells(cutoff=30)

        aggr_dict = {'e23Cux2': 'e23',
                     'e4Scnn1a': 'e4', 'e4Rorb': 'e4', 'e4other': 'e4', 'e4Nr5a1': 'e4',
                     'e5Rbp4': 'e5', 'e5noRbp4': 'e5',
                     'e6Ntsr1': 'e6',
                     'i1Htr3a': 'i1Htr3a',
                     'i23Htr3a': 'i23Htr3a',
                     'i23Pvalb': 'i23Pvalb',
                     'i23Sst': 'i23Sst',
                     'i4Htr3a': 'i4Htr3a',
                     'i4Pvalb': 'i4Pvalb',
                     'i4Sst': 'i4Sst',
                     'i5Htr3a': 'i5Htr3a',
                     'i5Pvalb': 'i5Pvalb',
                     'i5Sst': 'i5Sst',
                     'i6Htr3a': 'i6Htr3a',
                     'i6Pvalb': 'i6Pvalb',
                     'i6Sst': 'i6Sst',
                     }

        dataset.aggregate_cell_classes(aggr_dict)

        keepers = ['e23', 'e4', 'e5', 'e6', 'i1Htr3a', 'i23Htr3a', 'i23Pvalb',
                   'i23Sst', 'i4Htr3a', 'i4Pvalb', 'i4Sst', 'i5Pvalb',
                   'i5Sst', 'i6Htr3a', 'i6Pvalb', 'i6Sst']
        dataset.drop_other_classes(keepers)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, dataset.cell_type_labels

    def get_sampler(self, sparsity_thresh=0.75):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        # threshold
        #data = Batch.from_data_list(self[:])
        #x, y, batch = data.x, data.y, data.batch
        #batch = batch.to(x.device)
        #sparsity = global_add_pool((x.sum(axis=1) == 0).float(), batch) / 100
        #keep = torch.nonzero((sparsity<sparsity_thresh)).squeeze()
        #target = y[keep]
        
        target = self.target

        _, class_sample_count = torch.unique(target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(9 - torch.log(class_sample_count))))
        increase_factor = torch.clip(increase_factor, 0, 5)

        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3,4, 5]
        return SubsetRandomSampler(indices)


class V1Types13CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('v1_celltypes_13', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='v1', labels_col='13celltypes', force_process=False)

        # each sample must have at least 30 spikes
        dataset.drop_dead_cells(cutoff=30)

        # cell classes identified by Louis as not being too quiet
        keepers = ['e23', 'e6', 'i5Htr3a', 'i6Pvalb', 'i5Pvalb', 'i6Sst', 'i4Htr3a', 'i23Htr3a', 'e4', 'i1Htr3a',
                   'i4Sst', 'e5', 'i23Sst', 'i4Pvalb', 'i5Sst', 'i6Htr3a', 'i23Pvalb']
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
        print(increase_factor)
        # increase_factor = tensor([1., 1., 2., 1., 3., 3., 3., 2., 3., 3., 3., 2., 1.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)
        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


class V1Types11CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('v1_celltypes_11', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='v1', labels_col='11celltypes', force_process=False)

        # each sample must have at least 30 spikes
        dataset.drop_dead_cells(cutoff=30)

        # cell classes identified by Louis as not being too quiet
        keepers = ['Rorb', 'Rbp4', 'noRbp4', 'other', 'Ntsr1', 'Scnn1a', 'Pvalb', 'Cux2', 'Htr3a', 'Sst', 'Nr5a1']
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
        print(increase_factor)
        # increase_factor = tensor([1., 1., 1., 1., 2., 2., 1., 2., 2., 2., 1.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)
        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)



class V1Types4CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('v1_celltypes_4', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='v1', labels_col='4celltypes', force_process=False)

        # each sample must have at least 30 spikes
        dataset.drop_dead_cells(cutoff=30)

        # cell classes identified by Louis as not being too quiet
        keepers = ['Pvalb', 'Sst', 'e', 'Htr3a']
        dataset.drop_other_classes(classes_to_keep=keepers)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, keepers

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(10 - torch.log(class_sample_count))))
        print(increase_factor)
        #increase_factor=tensor([1., 2., 3., 3.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)
        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


# todo adapt to task/ dataset
class V1Layers5CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('v1_layers_5', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='v1', labels_col='5layers', force_process=False)

        # each sample must have at least 30 spikes
        dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['5', '23', '4', '1', '6']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(9 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor =
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)

    # todo adapt to task/ dataset


class V1Types2CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('v1_celltypes_2', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='v1', labels_col='2celltypes', force_process=False)

        # each sample must have at least 30 spikes
        dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['e', 'i']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(11 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([1., 2.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)

    # todo adapt to task/ dataset


class NeuropixelsBrainRegion4CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('neuropixels_brain_region_4', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='neuropixels', labels_col='brain_region', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['cortex', 'thalamus', 'hippocampus', 'midbrain']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(9 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([1., 3., 2., 2.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


# todo adapt to task/ dataset
class NeuropixelsNMBrainRegion4CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('neuropixels_nm_brain_region_4', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='neuropixels_nm', labels_col='brain_region',
                                  force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['cortex', 'thalamus', 'hippocampus', 'midbrain']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(9 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([1., 3., 2., 2.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


class NeuropixelsBrainStructure29CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('neuropixels_brain_structure_29', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='neuropixels', labels_col='brain_structure',
                                  force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = [
            'APN', 'CA1', 'CA3', 'DG', 'Eth', 'IGL', 'LGd', 'LGv', 'LP', 'MB',
            'MGd', 'MGv', 'NOT', 'PO', 'POL', 'ProS', 'SGN', 'SUB', 'TH',
            'VIS', 'VISal', 'VISam', 'VISl', 'VISli', 'VISmma', 'VISp',
            'VISpm', 'VISrl', 'VPM'
        ]
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(6 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([1., 1., 1., 1., 1., 3., 1., 1., 1., 1., 1., 3., 1., 3., 2., 2., 3., 1.,1., 3., 1., 2., 3., 2., 3., 2., 2., 3., 3.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


class NeuropixelsNMBrainStructure29CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('neuropixels_nm_brain_structure_29', root, split, random_seed, num_bins, force_process,
                         transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='neuropixels_nm', labels_col='brain_structure',
                                  force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = [
            'APN', 'CA1', 'CA3', 'DG', 'Eth', 'IGL', 'LGd', 'LGv', 'LP', 'MB',
            'MGd', 'MGv', 'NOT', 'PO', 'POL', 'ProS', 'SGN', 'SUB', 'TH',
            'VIS', 'VISal', 'VISam', 'VISl', 'VISli', 'VISmma', 'VISp',
            'VISpm', 'VISrl', 'VPM'
        ]
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(6 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([1., 1., 1., 1., 1., 3., 1., 1., 1., 1., 1., 3., 1., 3., 2., 2., 3., 1.,1., 3., 1., 2., 3., 2., 3., 2., 2., 3., 3.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


class NeuropixelsSubclass3CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('neuropixels_subclass_3', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='neuropixels', labels_col='subclass_unlabeled',
                                  force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['Sst', 'Vip', 'Pvalb']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(6 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([2., 2., 2.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


class NeuropixelsNMSubclass3CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('neuropixels_nm_subclass_3', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='neuropixels_nm', labels_col='subclass_unlabeled',
                                  force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['Sst', 'Vip', 'Pvalb']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(6 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([2., 2., 2.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


class NeuropixelsSubclass4CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('neuropixels_subclass_4', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='neuropixels', labels_col='subclass_unlabeled',
                                  force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['Sst', 'Vip', 'Pvalb', 'unlabeled']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(2, torch.floor(8.1 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([ 1., 16., 16., 16.]) need to implement random sampler
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


class NeuropixelsNMSubclass4CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('neuropixels_nm_subclass_4', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='neuropixels_nm', labels_col='subclass_unlabeled',
                                  force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['Sst', 'Vip', 'Pvalb', 'unlabeled']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(2, torch.floor(8.1 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([ 1., 16., 16., 16.]) need to implement random sampler
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


class CalciumBrainRegion6CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('calcium_brain_region_6', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='calcium', labels_col='brain_region', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['VISl', 'VISpm', 'VISrl', 'VISam', 'VISal', 'VISp']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(9 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([1., 1., 1., 1., 2., 2.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


class CalciumSubclass4CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('calcium_subclass_4', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='calcium', labels_col='subclass', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['Pvalb', 'Sst', 'Vip', 'e']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(10 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([1., 5., 5., 7.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


class CalciumSubclass13CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('calcium_subclass_13', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='calcium', labels_col='subclass_full', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['Rbp4', 'Slc17a7', 'Cux2', 'Fezf2', 'Ntsr1', 'Emx1', 'Sst', 'Tlx3', 'Scnn1a', 'Rorb', 'Nr5a1',
                       'Pvalb', 'Vip']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(8 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([2., 1., 1., 1., 1., 1., 2., 2., 2., 1., 2., 2., 3.])

        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)


class CalciumClass2CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('calcium_class_2', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='calcium', labels_col='class', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['e', 'i']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(10 - torch.log(class_sample_count))))
        print(increase_factor)
        # increase_factor = tensor([1., 3.])
        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices)
    

class CalciumNewTypes4CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('calcium_newtypes_4', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='calcium', labels_col='4newcelltypes', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['e', 'Vip', 'Sst', 'Pvalb']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(8 - torch.log(class_sample_count))))
        print(increase_factor)

        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices) 

class CalciumNMNewTypes4CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('calcium_nm_newtypes_4', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='calcium_nm', labels_col='4newcelltypes', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['e', 'Vip', 'Sst', 'Pvalb']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(8 - torch.log(class_sample_count))))
        print(increase_factor)

        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices) 

class CalciumNewTypes5CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('calcium_newtypes_5', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='calcium', labels_col='5newcelltypes', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['Cux2', 'Sst', 'Vip', 'Pvalb', 'Rorb']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(8 - torch.log(class_sample_count))))
        print(increase_factor)

        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices) 

class CalciumNMNewTypes5CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('calcium_nm_newtypes_5', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='calcium_nm', labels_col='5newcelltypes', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['Cux2', 'Sst', 'Vip', 'Pvalb', 'Rorb']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(8 - torch.log(class_sample_count))))
        print(increase_factor)

        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices) 

class CalciumNewTypes6CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('calcium_newtypes_6', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='calcium', labels_col='6newcelltypes', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['Cux2', 'Sst', 'Ntsr1', 'Vip', 'Pvalb', 'Rorb']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(8 - torch.log(class_sample_count))))
        print(increase_factor)

        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices) 

class CalciumNMNewTypes6CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('calcium_nm_newtypes_6', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='calcium_nm', labels_col='6newcelltypes', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['Cux2', 'Sst', 'Ntsr1', 'Vip', 'Pvalb', 'Rorb']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(8 - torch.log(class_sample_count))))
        print(increase_factor)

        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices) 
    
class CalciumNewTypes7CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('calcium_newtypes_7', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='calcium', labels_col='7newcelltypes', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['e4', 'Sst', 'Vip', 'Pvalb', 'e23', 'e6', 'e5']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(8.75 - torch.log(class_sample_count))))
        print(increase_factor)

        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices) 
    
class CalciumNMNewTypes7CellSets(CellSets):
    def __init__(self, root, split, random_seed, num_bins=128, force_process=False, transform=None):
        super().__init__('calcium_nm_newtypes_7', root, split, random_seed, num_bins, force_process, transform)

    def get_data(self, test_size=0.2, val_size=0.2):
        dataset = CellTypeDataset(self.root, data_source='calcium_nm', labels_col='7newcelltypes', force_process=False)

        # each sample must have at least 30 spikes
        # dataset.drop_dead_cells(cutoff=180)

        # cell classes identified by Louis as not being too quiet
        class_names = ['e4', 'Sst', 'Vip', 'Pvalb', 'e23', 'e6', 'e5']
        dataset.drop_other_classes(classes_to_keep=class_names)

        dataset.split_cell_train_val_test(test_size=test_size, val_size=val_size, seed=self.random_seed)
        return {'train': dataset.get_set('train'),
                'val': dataset.get_set('val'),
                'test': dataset.get_set('test')}, class_names

    def get_sampler(self):
        # weighted sampler
        # THIS WILL ONLY WORK FOR V1 DATA WITH 17 CLASSES, NEEDS TO BE ADJUSTED FOR OTHER TARGETS/DATASETS
        _, class_sample_count = torch.unique(self.target, return_counts=True)
        increase_factor = torch.floor(torch.pow(1.5, torch.floor(8.75 - torch.log(class_sample_count))))
        print(increase_factor)

        indices = []
        for cell_type, factor in enumerate(increase_factor.cpu()):
            cell_indices = torch.where(self.target == cell_type)[0]
            for _ in range(int(factor)):
                indices.append(cell_indices)

        indices = torch.cat(indices)  # [0, 1, 1, 2, 3, 4, 5]
        return SubsetRandomSampler(indices) 