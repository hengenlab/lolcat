q
To select splits:
![](docs/dataset_splits.png)


# Hyperparameter tuning
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 v1_hptune.py

# Training
CUDA_VISIBLE_DEVICES=2 python3 v1_train.py

# Open tensorboard
tensorboard --logdir=runs/ --bind_all --port=6010


# Data

```
raw
├── calcium_drifting_gratings
│   ├── calcium_cell_metadata.csv
│   ├── calcium_nodes.csv
│   ├── calcium_spikes_unaligned.csv
│   └── calcium_times_drifting_gratings_no_bads.csv
├── calcium_naturalistic_movies
│   ├── calcium_all_nm_spikes_unaligned.csv
│   ├── calcium_cell_metadata.csv
│   ├── calcium_nm_nodes.csv
│   └── calcium_times_natural_movie_three.csv
├── neuropixels_drifting_gratings
├── neuropixels_naturalistic_movies
└── v1_drifting_gratings
    ├── v1_gratings_order.txt
    ├── v1_nodes.csv
    └── v1_spikes.csv
```