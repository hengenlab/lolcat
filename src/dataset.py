import re
import os

import numpy as np
import pandas as pd


CELL_METADATA_FILENAME = {
    'v1': 'v1_nodes.csv',
    'neuropixels': 'neuropixels_all_nm_nodes.csv',
    'neuropixels_nm': 'neuropixels_all_nm_nodes.csv',
    'calcium': 'calcium_all_nm_nodes.csv',
    'calcium_nm': 'neuropixels_all_nm_nodes.csv'
}

SPIKE_FILENAME = {
    'v1': 'v1_spikes.csv',
    'neuropixels': 'neuropixels_spikes.csv',
    'neuropixels_nm': 'neuropixels_all_nm_spikes.csv',
    'calcium': 'calcium_spikes.csv',
    'calcium_nm': 'calcium_all_nm_spikes.csv'
}

GRATINGS_FILENAME = {
	'v1':'v1_gratings_order.txt',
	'neuropixels':'neuropixels_gratings_order.txt',
    'neuropixels_nm':'neuropixels_nm_order.txt',
	'calcium':'calcium_gratings_order.txt',
    'calcium_nm':'calcium_nm_order.txt'
}

def load_cell_metadata(root, *, data_source='v1', labels_col='pop_name'):
    try:
        filename = os.path.join(root, CELL_METADATA_FILENAME[data_source])
    except KeyError:
        KeyError('Data source ({}) does not exist.'.format(data_source))

    df = pd.read_csv(filename, sep=' ')

    # Get rid of the LIF neurons, keeping only biophysically realistic ones
    if (data_source == 'v1') & (labels_col == 'pop_name'):
        df = df[~df['pop_name'].str.startswith('LIF')]
        # df.sort_index()
   
    # Get cell ids
    cell_ids = df.id.to_numpy()

    # Get cell types
    cell_type_ids, cell_type_labels = pd.factorize(df[labels_col]) # get unique values and reverse lookup table
    return cell_ids, cell_type_ids, cell_type_labels.to_list()


def load_spike_data(root, *, data_source='v1', cell_ids):
    try:
        filename = os.path.join(root, SPIKE_FILENAME[data_source])
    except KeyError:
        KeyError('Data source ({}) does not exist.'.format(data_source))

    df = pd.read_csv(filename, sep=' ', usecols=['timestamps', 'node_ids'])  # only load the necessary columns
    df.timestamps = df.timestamps / 1000  # convert to seconds

    # perform inner join
    cell_series = pd.Series(cell_ids, name='node_ids')  # get index of cells of interest
    df = df.merge(cell_series, how='right', on='node_ids')  # do a one-to-many mapping so that cells that are not
                                                            # needed are filtered out and that cells that do not
                                                            # fire have associated nan row.
    assert df.node_ids.is_monotonic  # verify that nodes are sorted
    spiketimes = df.groupby(['node_ids'])['timestamps'].apply(np.array).to_numpy()  # group spike times for each
    # cell and create an array.
    return spiketimes


def load_trial_data(root, *, data_source='v1',):
    filename = os.path.join(root, GRATINGS_FILENAME[data_source])
    
    '''
    if data_source == 'calcium' or data_source.startswith('neuropixels'):
        print('({}) trial data not yet fully tested. Giving it a try...'.format(data_source))
    '''
    
    df = pd.read_csv(filename, engine='python', sep='  ', skiprows=12, usecols=[3], names=['filename'])

    # parse trial id
    # todo what is a trial id
    p = re.compile(r"trial_([0-9]+)")
    trial_id = df.filename.apply(lambda x: int(re.search(p, x).group(1))).to_list()

    # parse orientation
    if self.data_source in ['neuropixels','calcium','v1']:
        print('WARNING: trial id in neuropixels and calcium drifting gratings is dummy-valued')
        p = re.compile(r"ori([0-9]*\.?[0-9]+)")
        orientation = df.filename.apply(lambda x: float(re.search(p, x).group(1))).to_list()
        else: #does not apply for naturalistic movies getting idx of first frame instead 
            p = re.compile(r"_f([0-9]+)")
            orientation = df.filename.apply(lambda x: float(re.search(p, x).group(1))).to_list()

    trial_table = pd.DataFrame({'trial': trial_id, 'orientation': orientation})
    return trial_table
