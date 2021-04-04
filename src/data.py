import numpy as np
import pandas as pd
import os
import re
import pickle
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from functools import wraps

def run_once_property(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            instance = getattr(self, '_' + fn.__name__)
            return instance
        except AttributeError:
            instance = fn(self, *args, **kwargs)
            setattr(self, '_' + fn.__name__, instance)
            return instance
    return property(wrapper)

def requires(*attrs, error_msg=''):
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            if any((not hasattr(self, attr) for attr in attrs)):
                raise ValueError(error_msg)
            return fn(self, *args, **kwargs)
        return wrapper
    return decorator


class Dataset:
    trial_length = 3  # in seconds
    num_trials = 100
    raw_dir = 'raw/'
    processed_dir = 'processed/'
    processed_file = 'dataset.pkl'

    def __init__(self, root_dir, force_process=False):
        self.root_dir = root_dir

        ## Process data
        # check if already processed
        already_processed, filename = self._look_for_processed_file()

        if not(already_processed) or force_process:
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

    def _look_for_processed_file(self):
        filename = os.path.join(self.root_dir, self.processed_dir, self.processed_file)
        return os.path.exists(filename), filename

    def _load_cell_metadata(self):
        filename = os.path.join(self.root_dir, self.raw_dir, 'v1_nodes.csv')
        df = pd.read_csv(filename, sep=' ')

        # Get rid of the LIF neurons, keeping only biophysically realistic ones
        df = df[~df['pop_name'].str.startswith('LIF')]
        df.sort_index()

        cell_ids = df.id.to_numpy()
        # Get cell types
        cell_type_ids, cell_type_labels = pd.factorize(df.pop_name)  # get unique values and reverse lookup table
        return cell_ids, cell_type_ids, cell_type_labels.to_list()

    def _load_spike_data(self):
        filename = os.path.join(self.root_dir, self.raw_dir, 'spikes.csv')
        df = pd.read_csv(filename, sep=' ', usecols=['timestamps', 'node_ids'])  # only load the necessary columns
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
        df = pd.read_csv(filename, engine='python', sep='  ', skiprows=12, usecols=[3], names=['filename'])
        assert len(df) == self.num_trials

        # parse trial id
        # todo what is a trial id
        p = re.compile(r"trial_([0-9]+)")
        trial_id = df.filename.apply(lambda x: int(re.search(p, x).group(1))).to_list()

        # parse orientation
        p = re.compile(r"ori([0-9]*\.?[0-9]+)")
        orientation = df.filename.apply(lambda x: float(re.search(p, x).group(1))).to_list()

        trial_table = pd.DataFrame({'trial': trial_id, 'orientation': orientation})
        return trial_table

    def load(self, filename):
        with open(filename, 'rb') as input:
            processed = pickle.load(input)
        self.__dict__ = processed.__dict__.copy()  # doesn't need to be deep

    def save(self, filename):
        with open(filename, 'wb') as output:  # will overwrite it
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def split_cell_train_val_test(self, test_size=0.2, val_size=0.2, seed=1234):
        train_val_mask, test_mask = train_test_split(np.arange(len(self.cell_ids)), test_size=test_size, random_state=seed,
                                                     stratify=self.cell_type_ids)

        val_size = val_size / (1 - test_size) # adjust val size
        train_mask, val_mask = train_test_split(train_val_mask, test_size=val_size, random_state=seed,
                                                stratify=self.cell_type_ids[train_val_mask])
        self._cell_split_masks = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    def split_trial_train_val_test(self, test_size=0.2, val_size=0.2, temp=True, seed=1234):
        train_val_mask, test_mask = train_test_split(np.arange(len(self.trial_table)), test_size=test_size,
                                                     random_state=seed, shuffle=not temp)

        val_size = val_size / (1 - test_size)  # adjust val size
        train_mask, val_mask = train_test_split(train_val_mask, test_size=val_size, random_state=seed, shuffle=not temp)
        self._trial_split_masks = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    def set_bining_parameters(self, bin_size):
        self.bin_size = bin_size

    @run_once_property
    def cell_type_lookup_table(self):
        out = {}
        for split in ['train', 'val', 'test']:
            cell_type_lookup_table = []
            for cell_label_id in range(len(self.cell_type_labels)):
                mask  = self.cell_type_ids[self._cell_split_masks[split]] == cell_label_id
                cell_type_lookup_table.append(mask)
            out[split] = cell_type_lookup_table
        return out

    @requires('_cell_split_masks', '_trial_split_masks', error_msg='Split dataset first.')
    def sample(self, mode='train', sampler='U100', cell_random_seed=None, trial_random_seed=None, trial_id=None):
        assert mode in ['train', 'val', 'test']
        sampler_type = sampler[0]
        assert sampler_type == 'U'
        num_cells_per_class = int(sampler[1:])

        # select neurons
        if cell_random_seed:
            np.random.seed(cell_random_seed)
        split_mask = self._cell_split_masks[mode]
        split_lookup_table = self.cell_type_lookup_table[mode]
        select_mask = []
        for i in range(len(self.cell_type_labels)):
            select_mask.append(np.random.choice(split_mask, num_cells_per_class, replace=False,
                                                p=split_lookup_table[i]/split_lookup_table[i].sum()))
        select_mask = np.concatenate(select_mask)

        # select trial
        if trial_id is None:
            if trial_random_seed is not None:
                np.random.seed(trial_random_seed)
            trial_id = np.random.choice(self._trial_split_masks[mode])
        # todo else verify that this is not for training.
        trial_info = self.get_trial_info(trial_id)

        # bin data for selected samples
        start_time, end_time = trial_info['start_time'], trial_info['end_time']

        num_cells = len(select_mask)
        num_bins = int((end_time - start_time) / self.bin_size)

        bins = np.linspace(start_time, end_time, num_bins+1)  # arange doesn't work

        X = np.zeros((num_bins, num_cells))
        y = np.zeros((num_cells,))
        for i, cell in enumerate(select_mask):
            y[i] = self.cell_type_ids[cell]
            cell_spike_times = self.spike_times[cell]
            if np.isnan(cell_spike_times[0]):
                # cell that never fires
                # todo remove this from cell table
                continue
            X[:, i] = self._bin_data(cell_spike_times, bins)

        # additionnal metadata
        m = {'trial_id': trial_id, 'orientation': trial_info['orientation']}
        return X, y, m

    def get_trial_info(self, trial_id):
        start_time = trial_id * 3  # 3 seconds
        end_time = start_time + 3

        orientation = self.trial_table.loc[trial_id, 'orientation']
        return {'start_time': start_time, 'end_time': end_time, 'orientation': orientation}

    @requires('bin_size', error_msg='Set binning parameters first.')
    def _bin_data(self, spike_times, bins):
        firing_rates, _ = np.histogram(spike_times, bins)
        return firing_rates.astype(int)

    @property
    def num_cell_types(self):
        return len(self.cell_type_labels)

    def aggregate_cell_classes(self, aggr_dict):
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


if __name__ == '__main__':
    dataset = Dataset('./data')

    aggr_dict = {'e23Cux2': 'e23', 'i5Sst': 'i5Sst', 'i5Htr3a': 'i5Htr3a', 'e4Scnn1a': 'e4', 'e4Rorb': 'e4',
                 'e4other': 'e4', 'e4Nr5a1': 'e4', 'i6Htr3a': 'i6Htr3a', 'i6Sst': 'i6Sst', 'e6Ntsr1': 'e6',
                 'i23Pvalb': 'i23Pvalb', 'i23Htr3a': 'i23Htr3a', 'i1Htr3a': 'i1Htr3a', 'i4Sst': 'i4Sst', 'e5Rbp4': 'e5',
                 'e5noRbp4': 'e5', 'i23Sst': 'i23Sst', 'i4Htr3a': 'i4Htr3a', 'i6Pvalb': 'i6Pvalb', 'i5Pvalb': 'i5Pvalb',
                 'i4Pvalb': 'i4Pvalb'}

    print('Before aggregation: Number of cell types -', dataset.num_cell_types)
    dataset.aggregate_cell_classes(aggr_dict)

    print('After aggregation: Number of cell types -', dataset.num_cell_types)

