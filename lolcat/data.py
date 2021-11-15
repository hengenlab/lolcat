import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    processed_dir = 'processed/'

    @property
    def raw_dir(self):
        return os.path.join('raw/', self.data_source)

    @property
    def processed_file(self):
        return '{}_dataset.pkl'.format(self.data_source)

    def __init__(self, root_dir, data_source, force_process=False):
        self.root_dir = root_dir
        self.data_source = data_source

        # check if already processed
        already_processed, filename = self._look_for_processed_file()

        # if not processed or force_process
        if not (already_processed) or force_process:
            self.process()
            # pickle
            self.save(filename)
        else:
            print('Found processed pickle. Loading from %r.' % filename)
            self.load(filename)

    def process(self):
        raise NotImplementedError

    ##########################
    # LOADING PROCESSED DATA #
    ##########################
    def _look_for_processed_file(self):
        filename = os.path.join(self.root_dir, self.processed_dir, self.processed_file)
        return os.path.exists(filename), filename

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as output:  # will overwrite it
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as input:
            processed = pickle.load(input)
        self.__dict__ = processed.__dict__.copy()  # doesn't need to be deep

    #################
    # CHANGE LABELS #
    #################
    def drop_dead_cells(self, cutoff=1):
        # drop cells here
        # find neurons that satisfy the criteria in self.spike_times
        keep_mask = [((sts.size >= cutoff) and not (np.isnan(np.sum(sts)))) for sts in self.spike_times]

        self.cell_ids = self.cell_ids[keep_mask]
        self.spike_times = self.spike_times[keep_mask]
        self.cell_metadata = self.cell_metadata[keep_mask]

    def filter_cells(self, field, *, keep=None, drop=None):
        assert (keep is not None) != (drop is not None)

        # get field metadata
        field_id = self.cell_metadata_header.index(field)
        cell_labels = self.cell_metadata[:, field_id]

        label_names = keep or drop
        assert isinstance(label_names, list)

        mask = np.zeros_like(self.cell_ids, dtype=np.bool)
        for name in label_names:
            mask[cell_labels == name] = True

        if drop is not None:
            mask = np.logical_not(mask)

        self.cell_ids = self.cell_ids[mask]
        self.spike_times = self.spike_times[mask]
        self.cell_metadata = self.cell_metadata[mask]
        if hasattr(self, 'session_ids'):
            self.session_ids = self.session_ids[mask]

    ###########################
    # SPLIT TO TRAIN/VAL/TEST #
    ###########################
    def train_val_test_split(self, train_size=None, test_size=None, val_size=None, random_seed=1234, stratify_by=None):
        # parse sizes
        n_args = (train_size is not None) + (test_size is not None) + (val_size is not None)
        if n_args == 0:
            raise ValueError('Did not specify train/val/test split sizes.')
        elif n_args == 1:
            train_size = train_size or 0.
            test_size = test_size or 0.
            val_size = val_size or 0.
        elif n_args == 2:
            total_size = (train_size or 0.) + (test_size or 0.) + (val_size or 0.)
            assert 0. <= total_size <= 1.
            remainder_size = 1. - total_size
            train_size = train_size or remainder_size
            test_size = test_size or remainder_size
            val_size = val_size or remainder_size

        assert train_size + test_size + val_size == 1.

        # parse stratify argument
        if stratify_by is not None:
            assert isinstance(stratify_by, str)
            field_id = self.cell_metadata_header.index(stratify_by)
            labels = self.cell_metadata[:, field_id]
        else:
            labels = None

        # first, split into trainval and test
        if train_size + val_size == 1.:
            train_val_mask, test_mask = np.arange(len(self.cell_ids)), np.array([])
        elif train_size + val_size == 0.:
            train_val_mask, test_mask = np.array([]), np.arange(len(self.cell_ids))
        else:
            train_val_mask, test_mask = train_test_split(np.arange(len(self.cell_ids)),
                                                         test_size=test_size, random_state=random_seed, stratify=labels)

        # split trainval into train and val
        if val_size == 0.:
            train_mask, val_mask = train_val_mask, np.array([])
        elif train_size == 0.:
            train_mask, val_mask = np.array([]), train_val_mask
        else:
            val_size = val_size / (1. - test_size)  # adjust val size
            labels = labels[train_val_mask] if labels is not None else None

            train_mask, val_mask = train_test_split(train_val_mask,
                                                    test_size=val_size, random_state=random_seed, stratify=labels)

        self._cell_split = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    ##############
    # PROPERTIES #
    ##############
    def __len__(self):
        return len(self.cell_ids)

    ###################
    # Generate splits #
    ###################
    def _select_data(self, index, start_time, end_time):
        cell_spike_times = self.spike_times[index]
        if np.isnan(cell_spike_times[0]):
            print(index, 'never fires')
            # cell that never fires
            raise ValueError('Cell {} never fires.'.format(index))
        # only keep spike times between start_time and end_time
        cell_spike_times = cell_spike_times[(start_time <= cell_spike_times) & (cell_spike_times <= end_time)]
        cell_spike_times = np.sort(cell_spike_times)
        return cell_spike_times

    def get_data(self, split):
        if not hasattr(self, '_cell_split'):
            raise AssertionError('Split dataset first.')

        cell_ids = self._cell_split[split]
        data_list = []
        for cell_id in cell_ids:
            data_list.append(self[cell_id])
        return data_list

    def __getitem__(self, item):
        raise NotImplementedError


class V1Dataset(Dataset):
    name = 'v1'
    stimulus = 'drifting_gratings'

    trial_length = 3  # in seconds
    cell_metadata_filename = 'v1_nodes.csv'
    spike_times_filename = 'v1_spikes.csv'
    trial_metadata_filename = 'v1_gratings_order.txt'

    def __init__(self, root_dir, force_process=False):
        super(V1Dataset, self).__init__(root_dir, data_source='v1_drifting_gratings', force_process=force_process)

    def process(self):
        print('Processing data.')
        # load cell metadata
        self.cell_ids, self.cell_metadata, self.cell_metadata_header = self._load_cell_metadata()

        # load cell spike times
        self.spike_times = self._load_spike_data()

        # load trial metadata
        self.trial_metadata, self.trial_metadata_header = self._load_trial_data()

    ################
    # LOADING DATA #
    ################
    def _load_cell_metadata(self):
        filename = os.path.join(self.root_dir, self.raw_dir, self.cell_metadata_filename)
        df = pd.read_csv(filename, index_col='id')

        # Get rid of the LIF neurons, keeping only biophysically realistic ones
        df = df[~df['pop_name'].str.startswith('LIF')]

        # sort cells by id
        df.sort_index(inplace=True)

        # Get cell ids
        cell_ids = df.index.to_numpy()

        cell_metadata_header = df.columns.to_list()
        cell_metadata = df.to_numpy()

        return cell_ids, cell_metadata, cell_metadata_header

    def _load_spike_data(self):
        filename = os.path.join(self.root_dir, self.raw_dir, self.spike_times_filename)

        df = pd.read_csv(filename, sep=',', usecols=['timestamps', 'node_ids'])  # only load the necessary columns
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
        filename = os.path.join(self.root_dir, self.raw_dir, self.trial_metadata_filename)

        df = pd.read_csv(filename, engine='python', sep='  ', skiprows=12, usecols=[3], names=['filename'])

        # parse trial id
        # p = re.compile(r"trial_([0-9]+)")
        # trial_id = df.filename.apply(lambda x: int(re.search(p, x).group(1))).to_numpy()

        # parse orientation
        p = re.compile(r"ori([0-9]*\.?[0-9]+)")
        orientation = df.filename.apply(lambda x: float(re.search(p, x).group(1))).to_numpy()

        trial_metadata_header = ['orientation']
        trial_metadata = orientation[:, np.newaxis]
        return trial_metadata, trial_metadata_header

    ##############
    # PROPERTIES #
    ##############
    @property
    def num_trials(self):
        return self.trial_metadata.shape[0]

    ############
    # Get data #
    ############
    def __getitem__(self, item):
        data = defaultdict(list)

        # get trial windows
        for trial_id in np.arange(self.num_trials):
            # get start and end times of trial
            start_time = trial_id * self.trial_length  # 3 seconds
            end_time = start_time + self.trial_length

            # select spikes
            data['spikes'].append(self._select_data(item, start_time, end_time) - start_time)

        data = dict(data)

        # get full spike trains
        data['spike_blocks'] = [self._select_data(item, 0., self.trial_length*(self.num_trials+1))]

        # add trial metadata
        for i, header in enumerate(self.trial_metadata_header):
            data[header] = self.trial_metadata[:, i]

        # add cell metadata
        for i, header in enumerate(self.cell_metadata_header):
            data[header] = self.cell_metadata[item, i]
        return data


class CalciumDataset(Dataset):
    name = 'calcium'

    cell_metadata_filename = {
        'drifting_gratings': 'calcium_nodes.csv',
        'naturalistic_movies': 'calcium_nm_nodes.csv'
    }

    session_filename = {
        'drifting_gratings': 'calcium_times_drifting_gratings_no_bads.csv',
        'naturalistic_movies': 'calcium_times_natural_movie_three.csv',
    }

    spike_filename = {
        'drifting_gratings': 'calcium_spikes_unaligned.csv',
        'naturalistic_movies': 'calcium_all_nm_spikes_unaligned.csv'
    }

    metadata_filename = {
        'drifting_gratings': 'calcium_cell_metadata.csv',
        'naturalistic_movies': 'calcium_cell_metadata.csv',
    }

    def __init__(self, root_dir, stimulus, force_process=False):
        assert stimulus in ['drifting_gratings', 'naturalistic_movies']
        self.stimulus = stimulus

        data_source = self.name + '_' + self.stimulus
        super(CalciumDataset, self).__init__(root_dir, data_source=data_source, force_process=force_process)

    def process(self):
        print('Processing data.')
        # load cell metadata
        self.cell_ids, self.session_ids, self.session_names, \
        self.cell_metadata, self.cell_metadata_header = self._load_cell_metadata()

        # load extra cell metadata
        extra_metadata, extra_header = self._load_extra_metadata()
        if len(extra_header) != 0:
            self.cell_metadata = np.hstack([self.cell_metadata, extra_metadata])
            self.cell_metadata_header = self.cell_metadata_header + extra_header

        # load cell spike times
        self.spike_times = self._load_spike_data()

        # load session metadata
        self.session_metadata, self.trial_metadata, self.trial_metadata_header = self._load_session_metadata()

    ################
    # LOADING DATA #
    ################
    def _load_cell_metadata(self):
        filename = os.path.join(self.root_dir, self.raw_dir, self.cell_metadata_filename[self.stimulus])
        df = pd.read_csv(filename, index_col='id')

        if self.name == 'neuropixels':
            df = df[~df.isnull().any(axis=1)]

        # sort cells by id
        df.sort_index(inplace=True)

        # Get cell ids
        cell_ids = df.index.to_numpy()

        # Get session ids
        session_ids, session_names = pd.factorize(df['session_id'].astype(np.int))
        session_names = session_names.to_list()

        df = df.drop('session_id', axis=1)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        cell_metadata_header = df.columns.to_list()
        cell_metadata = df.to_numpy()

        return cell_ids, session_ids, session_names, cell_metadata, cell_metadata_header

    def _load_session_metadata(self):
        filename = os.path.join(self.root_dir, self.raw_dir, self.session_filename[self.stimulus])
        df = pd.read_csv(filename)

        df.start = df.start / 1000.  # convert to seconds
        df.end = df.end / 1000.  # convert to seconds

        session_metadata = [{} for _ in range(self.num_sessions)]
        trial_metadata = []
        for session_name in df.session_id.unique():
            df_session = df[df.session_id == session_name]
            df_session = df_session.drop('session_id', axis=1)

            if self.stimulus == 'drifting_gratings':
                start, end = np.array(df_session.start), np.array(df_session.end)
                metadata = df_session.to_numpy()
            elif self.stimulus == 'naturalistic_movies':
                start, end = np.array(df_session[0::90].start), np.array(df_session[89::90].end)
                metadata = df_session[0::90].to_numpy()

            trial_metadata.append(metadata)
            trial_metadata_header = df_session.columns.to_list()

            if self.name == 'calcium' and self.stimulus == 'drifting_gratings':
                start_, end_ = np.array(df_session.start), np.array(df_session.end)
                i1, i2 = np.where((start_[1:] - end_[:-1]) > 30)[0] + 1
                blocks = [(start_[0], end_[i1 - 1]), (start_[i1], end_[i2 - 1]), (start_[i2], end_[-1])]
            elif self.name == 'calcium' and self.stimulus == 'naturalistic_movies':
                start_, end_ = np.array(df_session.start), np.array(df_session.end)
                i = np.where((start_[1:] - end_[:-1]) > 30)[0] + 1
                blocks = [(start_[0], end_[i - 1]), (start_[i], end_[-1])]
            else:
                start_, end_ = np.array(df_session.start), np.array(df_session.end)
                blocks = [(start_[0], end_[-1])]
            try:
                session_id = self.session_names.index(session_name)
                session_metadata[session_id] = {'trials': (start, end), 'blocks': blocks}
            except:
                print('No cells from session: {}.'.format(session_name))
        return session_metadata, trial_metadata, trial_metadata_header

    def _load_spike_data(self):
        filename = os.path.join(self.root_dir, self.raw_dir, self.spike_filename[self.stimulus])
        df = pd.read_csv(filename, usecols=['timestamps', 'node_ids'])  # only load the necessary columns
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

    def _load_extra_metadata(self):
        filename = os.path.join(self.root_dir, self.raw_dir, self.metadata_filename[self.stimulus])
        df = pd.read_csv(filename)

        cell_series = pd.Series(self.cell_ids, name='dg_id')  # get index of cells of interest
        df = df.merge(cell_series, how='right', on='dg_id')  # do a one-to-many mapping so that cells that are not

        fields = ['area', 'tld1_name', 'tld2_name', 'tlr1_name', 'imaging_depth', 'osi_dg', 'dsi_dg', 'pref_dir_dg',
                  'pref_tf_dg', 'p_dg', 'g_dsi_dg', 'g_osi_dg', 'p_run_mod_dg', 'peak_dff_dg', 'reliability_dg',
                  'run_mod_dg', 'tfdi_dg']

        return df[fields].to_numpy(), fields

    ##############
    # PROPERTIES #
    ##############
    @property
    def num_sessions(self):
        return len(self.session_names) if self.session_names is not None else 1

    ############
    # Get data #
    ############
    def __getitem__(self, item):
        data = defaultdict(list)

        # get trial windows
        session_id = self.session_ids[item]
        session_metadata = self.session_metadata[session_id]

        for start_time, end_time in zip(*session_metadata['trials']):
            data['spikes'].append(self._select_data(item, start_time, end_time) - start_time)

        for start_time, end_time in session_metadata['blocks']:
            data['spike_blocks'].append(self._select_data(item, start_time, end_time) - start_time)

        data = dict(data)

        data['session_id'] = session_id

        # add trial metadata
        for i, header in enumerate(self.trial_metadata_header):
            data[header] = self.trial_metadata[session_id][:, i]

        # add cell metadata
        for i, header in enumerate(self.cell_metadata_header):
            data[header] = self.cell_metadata[item, i]
        return data


class NeuropixelsDataset(CalciumDataset):
    name = 'neuropixels'

    cell_metadata_filename = {
        'drifting_gratings':  'neuropixels_nodes.csv',
        'naturalistic_movies': 'neuropixels_all_nm_nodes.csv'
    }

    session_filename = {
        'drifting_gratings': 'neuropixels_times_drifting_gratings.csv',
        'naturalistic_movies': 'neuropixels_times_natural_movie_three_no_bads.csv',
    }

    spike_filename = {
        'drifting_gratings': 'neuropixels_spikes_unaligned.csv',
        'naturalistic_movies': 'neuropixels_all_nm_spikes_unaligned.csv'
    }

    def __init__(self, root_dir, stimulus, force_process=False):
        super(NeuropixelsDataset, self).__init__(root_dir, stimulus, force_process)

    def _load_extra_metadata(self):
        return [], []
