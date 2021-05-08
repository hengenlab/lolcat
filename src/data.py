import os
import re
import pickle
from collections import OrderedDict
from functools import wraps

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def run_once_property(fn):
    r"""Run fn once, when called the first time and then keep the result in memory."""
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
    r"""If attrs aren't defined, raises error."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            if any((not hasattr(self, attr) for attr in attrs)):
                raise ValueError(error_msg)
            return fn(self, *args, **kwargs)
        return wrapper
    return decorator


class Dataset:
    r"""
    """
    trial_length = 3  # in seconds
    num_trials = 100
    min
    raw_dir = 'raw/'
    processed_dir = 'processed/'
    processed_file = 'dataset.pkl'

    def __init__(self, root_dir, data_source='V1', labels_col = 'pop_name', force_process=False):
        self.root_dir = root_dir
        self.data_source = data_source

        # check if already processed
        already_processed, filename = self._look_for_processed_file()

        # if not processed or force_process
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
        
        self._trial_split = {'train': np.arange(len(self.trial_table)), 'val': np.arange(len(self.trial_table)), 'test': np.arange(len(self.trial_table))}
            
    def drop_dead_cells(self,cutoff=1):
        # drop cells here
        keep_mask = [True if ((sts.size >= cutoff) & (np.isnan(np.sum(sts)))==False) else False for sts in self.spike_times ]# find neurons that satisfy the criteria in self.spike_times
        self.spike_times = self.spike_times[keep_mask]
        self.cell_ids = self.cell_ids[keep_mask]
        self.cell_type_ids = self.cell_type_ids[keep_mask]
        
    def drop_other_classes(self,classes_to_keep):
        keep_mask = [True if self.cell_type_labels[cell_type] in classes_to_keep else False for cell_type in self.cell_type_ids]
        self.spike_times = self.spike_times[keep_mask]
        self.cell_ids = self.cell_ids[keep_mask]
        self.cell_type_ids = self.cell_type_ids[keep_mask]
        shift_dict = dict(zip(range(len(self.cell_type_labels)),range(len(self.cell_type_labels))))
        proceeding_numbers = []
        bad_nums = []
        for i in reversed(range(len(self.cell_type_labels))):
            if i in self.cell_type_ids:
                pass
            else:
                bad_nums.append(i)
                for pn in proceeding_numbers:
                    shift_dict[pn]-=1
            proceeding_numbers.append(i)
        self.cell_type_ids = np.asarray([int(shift_dict[i]) for i in self.cell_type_ids])
        self.cell_type_labels = np.asarray([l for i,l in enumerate(self.cell_type_labels) if i not in bad_nums])
                      

    def _look_for_processed_file(self):
        filename = os.path.join(self.root_dir, self.processed_dir, self.processed_file)
        return os.path.exists(filename), filename

    def _load_cell_metadata(self):
        data_source, labels_col = self.data_source, self.labels_col
        if data_source == 'v1':
            filename = os.path.join(self.root_dir, self.raw_dir, 'v1_nodes.csv')
        elif data_source == 'neuropixels_celltypes':
            filename = os.path.join(self.root_dir, self.raw_dir, 'neuropixels_celltypes_nodes.csv')
        elif data_source == 'neuropixels_regions':
            filename = os.path.join(self.root_dir, self.raw_dir, 'neuropixels_regions_nodes.csv')
        elif data_source == 'neuropixels_structures':
            filename = os.path.join(self.root_dir, self.raw_dir, 'neuropixels_structures_nodes.csv')
        
        elif data_source == 'calcium':
            filename = os.path.join(self.root_dir, self.raw_dir, 'calcium_nodes.csv')
        else:
            raise ValueError('Sampler %s does not exist.' % data_source)
            
        df = pd.read_csv(filename, sep=' ')

        # Get rid of the LIF neurons, keeping only biophysically realistic ones
        if (data_source == 'v1') & (labels_col == 'pop_name'):
            df = df[~df['pop_name'].str.startswith('LIF')]
            df.sort_index()

        cell_ids = df.id.to_numpy()
        # Get cell types
        cell_type_ids, cell_type_labels = pd.factorize(df[labels_col])  # get unique values and reverse lookup table
        return cell_ids, cell_type_ids, cell_type_labels.to_list()

    def _load_spike_data(self):
        data_source = self.data_source
        if data_source == 'v1':
            filename = os.path.join(self.root_dir, self.raw_dir, 'spikes.csv')
        elif data_source == 'neuropixels':
            filename = os.path.join(self.root_dir, self.raw_dir, 'neuropixels_spikes.csv')
        elif data_source == 'calcium':
            filename = os.path.join(self.root_dir, self.raw_dir, 'calcium_spikes.csv')
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
        data_source = self.data_source
        filename = os.path.join(self.root_dir, self.raw_dir, 'gratings_order.txt')
        if data_source in ['calcium','neuropixels']:
            print('{} trial data not yet implemented. Using V1 trial data.'.format(data_source))
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

    def save(self, filename):
        with open(filename, 'wb') as output:  # will overwrite it
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as input:
            processed = pickle.load(input)
        self.__dict__ = processed.__dict__.copy()  # doesn't need to be deep

    @property
    def num_cell_types(self):
        return len(self.cell_type_labels)

    def get_trial_info(self, trial_id):
        start_time = trial_id * 3  # 3 seconds
        end_time = start_time + (3*self.num_trials_in_window)
        orientation = self.trial_table.loc[trial_id, 'orientation']
        return {'start_time': start_time, 'end_time': end_time, 'orientation': orientation}

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
        print(set(self.cell_type_ids),self.cell_type_labels)

    def split_cell_train_val_test(self, test_size=0.2, val_size=0.2, seed=1234):
        train_val_mask, test_mask = train_test_split(np.arange(len(self.cell_ids)), test_size=test_size, random_state=seed,
                                                     stratify=self.cell_type_ids)

        val_size = val_size / (1 - test_size) # adjust val size
        
             
        train_mask, val_mask = train_test_split(train_val_mask, test_size=val_size, random_state=seed,
                                                stratify=self.cell_type_ids[train_val_mask])
        self._cell_split = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    def split_trial_train_val_test(self, test_size=0.2, val_size=0.2, temp=True, seed=1234):
        train_val_mask, test_mask = train_test_split(np.arange(len(self.trial_table)), test_size=test_size,
                                                     random_state=seed, shuffle=not temp)

        val_size = val_size / (1 - test_size)  # adjust val size
        train_mask, val_mask = train_test_split(train_val_mask, test_size=val_size, random_state=seed, shuffle=not temp)
        self._trial_split = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    def set_bining_parameters(self, bin_size):
        self.bin_size = bin_size

    @run_once_property
    def cell_type_lookup_table(self):
        out = {}
        for split in ['train', 'val', 'test']:
            cell_type_lookup_table = []
            for cell_label_id in range(len(self.cell_type_labels)):
                mask = self.cell_type_ids[self._cell_split[split]] == cell_label_id
                cell_type_lookup_table.append(mask)
            out[split] = cell_type_lookup_table
        return out

    def _uniform_sampler(self, mode, nbr_cells_per_class, random_seed=None):
        r""""""
        rng = np.random.default_rng(random_seed)
        split_mask = self._cell_split[mode]
        split_lookup_table = self.cell_type_lookup_table[mode]
        select_mask = []
        for i in range(len(self.cell_type_labels)):
            select_mask.append(rng.choice(split_mask, nbr_cells_per_class, replace=False,
                                          p=split_lookup_table[i]/split_lookup_table[i].sum()))
        return np.concatenate(select_mask)
    
    def _uniform_resampler(self, mode, nbr_cells_per_class, random_seed=None):
        r""""""
        rng = np.random.default_rng(random_seed)
        split_mask = self._cell_split[mode]
        split_lookup_table = self.cell_type_lookup_table[mode]
        select_mask = []
        for i in range(len(self.cell_type_labels)):
            if nbr_cells_per_class <= split_lookup_table[i].sum():
                select_mask.append(rng.choice(split_mask, nbr_cells_per_class, replace=False,
                                              p=split_lookup_table[i]/split_lookup_table[i].sum()))
            else:
                ss_mask = []
                for rs in range(nbr_cells_per_class//split_lookup_table[i].sum()):
                    ss_mask.append(rng.choice(split_mask, split_lookup_table[i].sum(), replace=False,
                                              p=split_lookup_table[i]/split_lookup_table[i].sum()))
                if nbr_cells_per_class%split_lookup_table[i].sum() != 0:
                    ss_mask.append(rng.choice(split_mask, nbr_cells_per_class%split_lookup_table[i].sum(), replace=False,
                                                  p=split_lookup_table[i]/split_lookup_table[i].sum()))
                select_mask.append(np.hstack(ss_mask))
            
        return np.concatenate(select_mask)
    
    
    def _balanced_sampler(self, mode, total_nbr_cells, random_seed=None):
        r""""""
        # todo use train_test_split with startify
        raise NotImplementedError

    @requires('bin_size', error_msg='Set binning parameters first.')
    def _bin_data(self, select_mask, start_time, end_time):
        num_cells = len(select_mask)
        num_bins = int((end_time - start_time) / self.bin_size)

        bins = np.linspace(start_time, end_time, num_bins + 1)  # arange doesn't work

        X = np.zeros((num_bins, num_cells))
        for i, cell in enumerate(select_mask):
            cell_spike_times = self.spike_times[cell]
            if np.isnan(cell_spike_times[0]):
                # cell that never fires
                # todo remove this from cell table
                continue
            firing_rates, _ = np.histogram(cell_spike_times, bins)
            X[:, i] = firing_rates.astype(int)
        return X

    def _select_data(self, select_mask, start_time, end_time):
        X = []
        for i, cell in enumerate(select_mask):
            cell_spike_times = self.spike_times[cell]
            if np.isnan(cell_spike_times[0]):
                # cell that never fires
                # todo remove this from cell table
                X.append(np.array([]))
                continue
            # only keep spike times between start_time and end_time
            cell_spike_times = cell_spike_times[(start_time <= cell_spike_times) & (cell_spike_times <= end_time)]
            cell_spike_times = np.sort(cell_spike_times)
            X.append(cell_spike_times)
        X = np.array(X)
        return X

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

    def get_trials(self, mode=None):
        if mode is None:
            return np.arrange(len(self.trial_table))
        else:
            cell_mode, time_mode = self.parse_mode(mode)
            return self._trial_split[time_mode]

    @requires('_cell_split', '_trial_split', error_msg='Split dataset first.')
    def sample(self, mode='train', sampler='U100', transform=None,
               cell_random_seed=None, trial_random_seed=None, trial_id=None, remove_silents=False, preselected_mask=None):

        cell_mode, time_mode = self.parse_mode(mode)

        ### Sample population
        # parse sampler information
        sampler_type = sampler[0]
        sampler_param = int(sampler[1:])
        if sampler_type == 'U':
            sampler = self._uniform_sampler
        elif sampler_type == 'B':
            sampler = self._balanced_sampler
        elif sampler_type == 'R':
            sampler = self._uniform_resampler
        else:
            raise ValueError('Sampler %s does not exist.' % sampler_type)

        # sample cells
        # todo probably want to use fine cell labels when doing this.
        #  If the number of cell types is reduced to 2 for example
        if preselected_mask is None:
            select_mask = sampler(cell_mode, sampler_param, random_seed=cell_random_seed)
        else:
            select_mask = preselected_mask
        
        # select trial
        if trial_id is None:
            # then random pick one
            rng = np.random.default_rng(trial_random_seed)
            trial_id = rng.choice(self._trial_split[time_mode])
        else:
            # raise error if trial_id is from a different subset
            assert trial_id in self._trial_split[time_mode]
        trial_info = self.get_trial_info(trial_id)

        # todo currently the trials are forced to be split into blocks
        start_time, end_time = trial_info['start_time'], trial_info['end_time']

        ### Transform data
        if transform is None:
            # X will be an array of arrays, each row will contain a vector that will have a dynamic shape
            X = self._select_data(select_mask, start_time, end_time)
        elif transform == 'firing_rate':
            # X will be a square matrix with shape: (num_bins, num_cells)
            X = self._bin_data(select_mask, start_time, end_time)
        elif transform == 'interspike_interval':
            # X will be an array of arrays, each row will contain a vector that will have a dynamic shape
            X = self._select_data(select_mask, start_time, end_time)
            # just compute diff
            X = np.array([np.diff(x) for x in X])
        elif transform == 'log_interspike_interval':
            # X will be an array of arrays, each row will contain a vector that will have a dynamic shape
            X = self._select_data(select_mask, start_time, end_time)
            # just compute diff
            X = np.array([np.diff(x) for x in X])
            X = np.array([np.log(x) for x in X])
            X_mins = np.array([np.min(x) for x in X if len(x) > 0])
            X_maxes = np.array([np.max(x) for x in X if len(x) > 0])
        else:
            raise ValueError('Transform method %r does not exist' %transform)
       
        # Get labels
        y = self.cell_type_ids[select_mask]

        # Get any additional metadata
        m = {'trial_id': trial_id, 'orientation': trial_info['orientation']}
        return X, y, m


if __name__ == '__main__':
    # todo add examples here
    dataset = Dataset('./data')

    aggr_dict = {'e23Cux2': 'e23', 'i5Sst': 'i5Sst', 'i5Htr3a': 'i5Htr3a', 'e4Scnn1a': 'e4', 'e4Rorb': 'e4',
                 'e4other': 'e4', 'e4Nr5a1': 'e4', 'i6Htr3a': 'i6Htr3a', 'i6Sst': 'i6Sst', 'e6Ntsr1': 'e6',
                 'i23Pvalb': 'i23Pvalb', 'i23Htr3a': 'i23Htr3a', 'i1Htr3a': 'i1Htr3a', 'i4Sst': 'i4Sst', 'e5Rbp4': 'e5',
                 'e5noRbp4': 'e5', 'i23Sst': 'i23Sst', 'i4Htr3a': 'i4Htr3a', 'i6Pvalb': 'i6Pvalb', 'i5Pvalb': 'i5Pvalb',
                 'i4Pvalb': 'i4Pvalb'}

    print('Before aggregation: Number of cell types -', dataset.num_cell_types)
    dataset.aggregate_cell_classes(aggr_dict)

    print('After aggregation: Number of cell types -', dataset.num_cell_types)

