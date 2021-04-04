# label_dict is a dictionary of all the label sets available, for V1 and
# Neuropixels datasets
neuropixels_brain_structures = {
    'cortex': [
        'VISp', 'VISl', 'VISrl', 'VISam', 'VISpm', 'VIS', 'VISal', 'VISmma',
        'VISmmp', 'VISli'
    ],
    'thalamus': [
        'LGd', 'LD', 'LP', 'VPM', 'TH', 'MGm', 'MGv', 'MGd', 'PO', 'LGv', 'VL',
        'VPL', 'POL', 'Eth', 'PoT', 'PP', 'PIL', 'IntG', 'IGL', 'SGN',
        'PF', 'RT'
    ],
    'hippocampus': [
        'CA1', 'CA2', 'CA3', 'DG', 'SUB', 'POST', 'PRE', 'ProS', 'HPF'
    ],
    'midbrain': [
        'MB', 'SCig', 'SCiw', 'SCsg', 'SCzo', 'PPT', 'APN', 'NOT', 'MRN',
        'OP', 'LT', 'RPF', 'CP'
    ],
    'grey': [
        'grey'
    ],
    'other': [
        'ZI'
    ]
}

# This is for assigning each Neuropixels unit to a brain region, knowing the
# brain structure (which is the ecephys_brain_structure ; brain regions are not
# available through the allensdk, hence the necessity of such a dict)
structure_to_region = {
    structure: region
    for region in neuropixels_brain_structures.keys()
    for structure in [
        structure
        for structure in neuropixels_brain_structures[region]
    ]
}


label_dict = {
    'v1': {
        'e_vs_i': [
            'e', 'i'
        ],
        'cell_type': [
            'e', 'Sst', 'Htr3a', 'Pvalb'
        ],
        'cell_type_full': [
            'e5Rbp4', 'Cux2', 'Pvalb', 'Scnn1a', 'Htr3a', 'Rorb', 'other', 'Sst',
            'Nr5a1', 'e5noRbp4', 'Ntsr1'
        ],
        'cell_type_i' : [
            'Sst', 'Htr3a', 'Pvalb'
        ],
        'cell_class': [
            'e23', 'i5Sst', 'i5Htr3a', 'e4', 'i6Htr3a', 'i6Sst', 'e6', 'i23Pvalb',
            'i23Htr3a', 'i1Htr3a', 'i4Sst', 'e5', 'i23Sst', 'i4Htr3a', 'i6Pvalb',
            'i5Pvalb', 'i4Pvalb'
        ],
        'cell_class_i': [
            'i5Sst', 'i5Htr3a', 'i6Htr3a', 'i6Sst', 'i23Pvalb', 'i23Htr3a',
            'i1Htr3a', 'i4Sst', 'i23Sst', 'i4Htr3a', 'i6Pvalb','i5Pvalb',
            'i4Pvalb'
        ],
        'cell_class_full': [
            'e23Cux2', 'i5Sst', 'i5Htr3a', 'e4Scnn1a', 'e4Rorb', 'e4other',
            'e4Nr5a1', 'i6Htr3a', 'i6Sst', 'e6Ntsr1', 'i23Pvalb', 'i23Htr3a',
            'i1Htr3a', 'i4Sst', 'e5Rbp4', 'e5noRbp4', 'i23Sst', 'i4Htr3a',
            'i6Pvalb', 'i5Pvalb', 'i4Pvalb'
        ],
        'layer': [
            '1', '23', '4', '5', '6'
        ]
    },
    'neuropixels': {
        'cell_type': [
            'Sst', 'Vip', 'Pvalb'
        ],
        'brain_structure': [
            'VISl', 'VISpm', 'VISrl', 'VISam', 'VISal', 'VISp', 'VIS',
            'CA1', 'DG', 'SUB', 'CA3',
            'APN',
            'LP', 'LGd'
        ],
        'brain_structure_full': [
            'VISl', 'VISpm', 'VISrl', 'VISam', 'VISal', 'VISp' 'VIS', 'VISmma',
            'CA1', 'DG', 'SUB', 'CA3', 'ProS',
            'APN', 'MB',
            'LP', 'LGd', 'PO', 'VPM', 'Eth', 'SGN', 'MGv', 'TH', 'LGv', 'MGd', 'POL'
        ],
        'brain_region': [
            'cortex', 'thalamus', 'hippocampus', 'midbrain'
        ],
        'cortex': [
            'VISl', 'VISpm', 'VISrl', 'VISam', 'VISal', 'VISp'
        ],
        'hippocampus': [
            'CA1', 'DG', 'SUB', 'CA3'
        ],
        'thalamus': [
            'LP', 'LGd', 'PO', 'VPM'
        ],
        'layer': [
            '2/3', '4', '5', '6'
        ]
    },
    'calcium': {
        'cell_type' : [
            'e', 'Sst', 'Vip', 'Pvalb'
        ],
        'cell_type_full' : [
            'Fezf2', 'Rorb', 'Slc17a7', 'Rbp4', 'Emx1', 'Tlx3',
            'Ntsr1', 'Cux2', 'Nr5a1', 'Scnn1a', 'Pvalb', 'Sst', 'Vip'
        ],
        'cell_type_excitatory' : [
            'Fezf2', 'Rorb', 'Slc17a7', 'Rbp4', 'Emx1', 'Tlx3',
            'Ntsr1', 'Cux2', 'Nr5a1', 'Scnn1a'
            ],
        'cell_type_inhibitory' : [
            'Pvalb', 'Sst', 'Vip'
            ],
        'brain_structure' : [
            'VISl', 'VISpm', 'VISrl', 'VISam', 'VISal', 'VISp'
        ],
    }
}

# This dictionary maps stimuli to their actual name for each data set
stimulus_dict = {
   'v1': {
        'DG' : 'driftingGratings',
        'NM' : 'naturalMovie'
    },
    'neuropixels' : {
        'DG' : 'drifting_gratings',
        'SG' : 'static_gratings',
        'NM' : 'natural_movie_one',
        'NS' : 'natural_scenes',
    },
    'calcium' :  {
        'DG' : 'drifting_gratings',
        'SG' : 'static_gratings',
        'NM' : 'natural_movie_one',
        'NS' : 'natural_images',
    }
}

neuropixels_sessions = [
 'session_750749662',
 'session_715093703',
 'session_719161530',
 'session_721123822',
 'session_732592105',
 'session_737581020',
 'session_739448407',
 'session_742951821',
 'session_743475441',
 'session_744228101',
 'session_746083955',
 'session_750332458',
 'session_751348571',
 'session_754312389',
 'session_754829445',
 'session_755434585',
 'session_756029989',
 'session_757216464',
 'session_757970808',
 'session_758798717',
 'session_759883607',
 'session_760345702',
 'session_760693773',
 'session_761418226',
 'session_762120172',
 'session_762602078',
 'session_763673393',
 'session_773418906',
 'session_791319847',
 'session_797828357',
 'session_798911424',
 'session_799864342']