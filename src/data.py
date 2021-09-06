import os
import pickle
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import requires


class Dataset:
    r"""
    v1: pop_name
    neuropixels: brain_region, brain_structure
    """
    trial_length = 3  # in seconds
    num_trials = 100
    raw_dir = 'raw/'
    processed_dir = 'processed/'

    @property
    def processed_file(self):
        return '{}_dataset-labels_{}.pkl'.format(self.data_source, self.labels_col)

    @property
    def csv_sep(self):
        if self.data_source == 'v1':
            sep = ' '
        else:
            sep = ','
        return sep

    def __init__(self, root_dir, data_source='v1', force_process=False, labels_col='pop_name'):
        self.root_dir = root_dir
        self.data_source = data_source
        self.labels_col = labels_col

        # check if already processed
        already_processed, filename = self._look_for_processed_file()

        # if not processed or force_process
        if not (already_processed) or force_process:
            print('Processing data.')
            # process
            self.cell_ids, self.cell_type_ids, self.cell_type_labels = self._load_cell_metadata()
            self.spike_times = self._load_spike_data()
            self.trial_table = self._load_trial_data()

            # pickle
            self.save(filename)
        else:
            print('Found processed pickle. Loading from %r.' % filename)
            self.load(filename)

        self._trial_split = {'train': np.arange(len(self.trial_table)),
                             'val': np.arange(len(self.trial_table)),
                             'test': np.arange(len(self.trial_table))}

    ##########################
    # LOADING PROCESSED DATA #
    ##########################
    def _look_for_processed_file(self):
        filename = os.path.join(self.root_dir, self.processed_dir, self.processed_file)
        return os.path.exists(filename), filename

    def save(self, filename):
        with open(filename, 'wb') as output:  # will overwrite it
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as input:
            processed = pickle.load(input)
        self.__dict__ = processed.__dict__.copy()  # doesn't need to be deep

    ################
    # LOADING DATA #
    ################
    def _load_cell_metadata(self):
        CELL_METADATA_FILENAME = {
            'v1': 'v1_nodes.csv',
            'neuropixels': 'neuropixels_nodes.csv',
            'calcium': 'calcium_nodes.csv',
        }

        try:
            filename = os.path.join(self.root_dir, self.raw_dir, CELL_METADATA_FILENAME[self.data_source])
        except KeyError:
            KeyError('Data source ({}) does not exist.'.format(self.data_source))

        df = pd.read_csv(filename, sep=self.csv_sep, index_col='id')

        # Get rid of the LIF neurons, keeping only biophysically realistic ones
        if (self.data_source == 'v1') & (self.labels_col == 'pop_name'):
            df = df[~df['pop_name'].str.startswith('LIF')]

        # sort cells by id
        df.sort_index(inplace=True)

        # Get cell ids
        cell_ids = df.index.to_numpy()

        # Get cell types
        cell_type_ids, cell_type_labels = pd.factorize(df[self.labels_col])  # get unique values and reverse lookup table
        return cell_ids, cell_type_ids.astype(np.int), cell_type_labels.to_list()

    def _load_spike_data(self):
        SPIKE_FILENAME = {
            'v1': 'spikes.csv',
            'neuropixels': 'neuropixels_spikes.csv',
            'calcium': 'calcium_spikes.csv'
        }

        try:
            filename = os.path.join(self.root_dir, self.raw_dir, SPIKE_FILENAME[self.data_source])
        except KeyError:
            KeyError('Data source ({}) does not exist.'.format(self.data_source))

        df = pd.read_csv(filename, sep=self.csv_sep, usecols=['timestamps', 'node_ids'])  # only load the necessary columns
        df.timestamps = df.timestamps / 1000  # convert to seconds

        # perform inner join
        cell_series = pd.Series(self.cell_ids, name='node_ids')  # get index of cells of interest
        df = df.merge(cell_series, how='right', on='node_ids')  # do a one-to-many mapping so that cells that are not
        # needed are filtered out and that cells that do not
        # fire have associated nan row.
        assert df.node_ids.is_monotonic  # verify that nodes are sorted
        spiketimes = df.groupby(['node_ids'])['timestamps'].apply(np.array).to_numpy()  # group spike times for each
        # cell and create an array.
        return spiketimes

    def _load_trial_data(self):
        filename = os.path.join(self.root_dir, self.raw_dir, 'gratings_order.txt')

        if self.data_source == 'calcium' or self.data_source.startswith('neuropixels'):
            print('({}) trial data not yet implemented. Using V1 trial data.'.format(self.data_source))
        df = pd.read_csv(filename, engine='python', sep='  ', skiprows=12, usecols=[3], names=['filename'])
        assert len(df) == self.num_trials

        # parse trial id
        p = re.compile(r"trial_([0-9]+)")
        trial_id = df.filename.apply(lambda x: int(re.search(p, x).group(1))).to_list()

        # parse orientation
        p = re.compile(r"ori([0-9]*\.?[0-9]+)")
        orientation = df.filename.apply(lambda x: float(re.search(p, x).group(1))).to_list()

        trial_table = pd.DataFrame({'trial': trial_id, 'orientation': orientation})
        return trial_table

    #################
    # CHANGE LABELS #
    #################
    def aggregate_cell_classes(self, aggr_dict):
        r"""Groups cell sub-classes into aggregates. The :obj:`aggr_dict` defines where each cell class is mapped to.

        Will handle changing the labels for all neurons in the data.
        ..note ::
            Use to map all 21 cell types to 2 classes for example (excitatory and inhibitory).
        """
        aggregates = OrderedDict()
        aggregation_map = []
        for cell_type in self.cell_type_labels:
            cell_group = aggr_dict[cell_type]
            if not cell_group in aggregates:
                aggregates[cell_group] = len(aggregates)
            aggregation_map.append(aggregates[cell_group])
        aggregation_map = np.array(aggregation_map)

        new_cell_type_labels = list(aggregates.keys())
        new_cell_type_ids = aggregation_map[self.cell_type_ids]
        self.cell_type_labels, self.cell_type_ids = new_cell_type_labels, new_cell_type_ids

    def drop_dead_cells(self, cutoff=1):
        # drop cells here
        # find neurons that satisfy the criteria in self.spike_times
        keep_mask = [((sts.size >= cutoff) and not (np.isnan(np.sum(sts)))) for sts in self.spike_times]

        self.spike_times = self.spike_times[keep_mask]
        self.cell_ids = self.cell_ids[keep_mask]
        self.cell_type_ids = self.cell_type_ids[keep_mask]

        unique_cell_type_ids = np.unique(self.cell_type_ids)
        assert unique_cell_type_ids.shape == unique_cell_type_ids[-1] + 1, "One cell class no longer has cells."

    def drop_other_classes(self, classes_to_keep):
        keep_mask = [self.cell_type_labels[cell_type] in classes_to_keep for cell_type in self.cell_type_ids]
        keep_mask = np.array(keep_mask).astype(np.bool)

        self.spike_times = self.spike_times[keep_mask]
        self.cell_ids = self.cell_ids[keep_mask]

        # relabel map
        relabel_mask = [cell_type in classes_to_keep for cell_type in self.cell_type_labels]
        relabel_mask = np.array(relabel_mask).astype(np.bool)
        relabel_map = np.zeros(len(relabel_mask))
        relabel_map[relabel_mask] = np.arange(relabel_mask.sum()).astype(int)

        self.cell_type_ids = relabel_map[self.cell_type_ids[keep_mask]]
        self.cell_type_labels = [label for label, m in zip(self.cell_type_labels, keep_mask) if m]

    ###########################
    # SPLIT TO TRAIN/VAL/TEST #
    ###########################
    def split_cell_train_val_test(self, test_size=0.2, val_size=0.2, seed=1234):
        train_val_mask, test_mask = train_test_split(np.arange(len(self.cell_ids)),
                                                     test_size=test_size,
                                                     random_state=seed,
                                                     stratify=self.cell_type_ids)

        val_size = val_size / (1 - test_size)  # adjust val size

        train_mask, val_mask = train_test_split(train_val_mask, test_size=val_size, random_state=seed,
                                                stratify=self.cell_type_ids[train_val_mask])
        self._cell_split = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    def split_trial_train_val_test(self, test_size=0.2, val_size=0.2, temp=True, seed=1234):
        if not temp: raise NotImplementedError
        train_val_mask, test_mask = train_test_split(np.arange(len(self.trial_table)), test_size=test_size,
                                                     random_state=seed, shuffle=not temp)

        val_size = val_size / (1 - test_size)  # adjust val size
        train_mask, val_mask = train_test_split(train_val_mask, test_size=val_size, random_state=seed, shuffle=not temp)
        self._trial_split = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    @staticmethod
    def parse_mode(mode):
        assert mode in ['train', 'val', 'test', 'val_time', 'test_time', 'val_cell', 'test_cell']
        l = mode.split('_')
        if len(l) == 1:
            cell_mode = time_mode = mode
        else:
            if l[1] == 'time':
                cell_mode = 'train'
                time_mode = l[0]
            else:  # l[1] == 'cell':
                cell_mode = l[0]
                time_mode = 'train'
        return cell_mode, time_mode

    ##############
    # PROPERTIES #
    ##############
    @property
    def num_cell_types(self):
        return len(self.cell_type_labels)

    ###################
    # Generate splits #
    ###################
    def _select_data(self, select_mask, start_time, end_time):
        X = []
        for i, cell in enumerate(select_mask):
            cell_spike_times = self.spike_times[cell]
            if np.isnan(cell_spike_times[0]):
                # cell that never fires
                raise ValueError
            # only keep spike times between start_time and end_time
            cell_spike_times = cell_spike_times[(start_time <= cell_spike_times) & (cell_spike_times <= end_time)]
            cell_spike_times = np.sort(cell_spike_times)
            X.append(cell_spike_times)
        X = np.array(X)
        return X

    @requires('_cell_split', '_trial_split', error_msg='Split dataset first.')
    def get_set(self, mode, num_trials_in_window=1, window_stride=1):
        # parse mode
        cell_mode, time_mode = self.parse_mode(mode)

        # get cells in cell_mode
        cell_ids = self._cell_split[cell_mode]

        X, y = [], []

        # iterate over trials and collect features
        trial_iterator = self._trial_split[time_mode]
        if num_trials_in_window > 1:
            trial_iterator = trial_iterator[:-num_trials_in_window + 1:window_stride]

        for trial_id in trial_iterator:
            # get time window
            start_time = trial_id * self.trial_length  # 3 seconds
            end_time = start_time + (self.trial_length * num_trials_in_window)

            # select data
            X.append(self._select_data(cell_ids, start_time, end_time) - start_time)
            y.append(self.cell_type_ids[cell_ids])

        X = np.concatenate(X)
        y = np.concatenate(y)

        return X, y


if __name__ == '__main__':
    # NOTE: THIS ISN'T MEANT TO BE USED, JUST AN EXAMPLE OF THE FLOW
    dataset = Dataset('./data')

    aggr_dict = {'e23Cux2': 'e23', 'i5Sst': 'i5Sst', 'i5Htr3a': 'i5Htr3a', 'e4Scnn1a': 'e4', 'e4Rorb': 'e4',
                 'e4other': 'e4', 'e4Nr5a1': 'e4', 'i6Htr3a': 'i6Htr3a', 'i6Sst': 'i6Sst', 'e6Ntsr1': 'e6',
                 'i23Pvalb': 'i23Pvalb', 'i23Htr3a': 'i23Htr3a', 'i1Htr3a': 'i1Htr3a', 'i4Sst': 'i4Sst', 'e5Rbp4': 'e5',
                 'e5noRbp4': 'e5', 'i23Sst': 'i23Sst', 'i4Htr3a': 'i4Htr3a', 'i6Pvalb': 'i6Pvalb', 'i5Pvalb': 'i5Pvalb',
                 'i4Pvalb': 'i4Pvalb'}

    print('Before aggregation: Number of cell types -', dataset.num_cell_types)
    dataset.aggregate_cell_classes(aggr_dict)

    print('After aggregation: Number of cell types -', dataset.num_cell_types)

    dataset.drop_dead_cells()
    dataset.split_cell_train_val_test(test_size=0.8, val_size=0.1)
    dataset.split_trial_train_val_test(test_size=0.8, val_size=0.1)

    fr_transform = FiringRates(window_size=3, bin_size=0.5)
    isi_transform = ISIDistribution(bins=10, min_isi=0, max_isi=0.4)
    fr_isi_transform = ConcatFeats(fr_transform, isi_transform)

    X_train, y_train = dataset.get_set('train', transform=fr_isi_transform)

    # drop rows for which cell is not very active
    mask = X_train[:, :6].sum(axis=1) > threshold
    X_train, y_train = X_train[mask], y_train[mask]

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    train_dataset = TensorDataset(X_train, y_train)

    # sampler = # https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, sampler=sampler)

    X_val, y_val = dataset.get_set('val', transform=fr_isi_transform)  # do not use accuracy, use F1-score
