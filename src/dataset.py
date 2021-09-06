import re
import os

import numpy as np
import pandas as pd


CELL_METADATA_FILENAME = {
    'v1': 'v1_nodes.csv',
    'neuropixels': 'neuropixels_nodes.csv',
    'calcium': 'calcium_nodes.csv',
}

SPIKE_FILENAME = {
    'v1': 'v1_spikes.csv',
    'neuropixels': 'neuropixels_spikes.csv',
    'calcium': 'calcium_spikes.csv',
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
    filename = os.path.join(root, 'gratings_order.txt')

    if data_source == 'calcium' or data_source.startswith('neuropixels'):
        print('({}) trial data not yet implemented. Using V1 trial data.'.format(data_source))

    df = pd.read_csv(filename, engine='python', sep='  ', skiprows=12, usecols=[3], names=['filename'])

    # parse trial id
    # todo what is a trial id
    p = re.compile(r"trial_([0-9]+)")
    trial_id = df.filename.apply(lambda x: int(re.search(p, x).group(1))).to_list()

    # parse orientation
    p = re.compile(r"ori([0-9]*\.?[0-9]+)")
    orientation = df.filename.apply(lambda x: float(re.search(p, x).group(1))).to_list()

    trial_table = pd.DataFrame({'trial': trial_id, 'orientation': orientation})
    return trial_table
