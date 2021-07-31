#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:41:46 2020

@author: louis
"""
from comet_ml import Experiment

import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import xgboost as xgb
import itertools
import argparse
import random
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import v_measure_score, f1_score

import functools
from smart_open import open
from boto3.session import Session
from botocore.config import Config

import warnings
warnings.filterwarnings("ignore") 

# Hyperparameter tuning function
class weightedRandomKFold():
    def __init__(self, n_splits, n_repeats=1):
        self.n_splits = n_splits
        self.n_repeats = n_repeats

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits*self.n_repeats

    def split(self, X, y, groups=None):
        splits = np.array_split(np.random.choice(len(X), len(X),replace=False), 5)
        train, test = [], []
        for repeat in range(self.n_repeats):
            for idx in range(len(splits)):
                train_indices = np.delete(splits, idx, axis=0)
                train_indices = np.hstack(train_indices)
                counter = Counter(y[train_indices])
                weights = [len(train_indices) / counter[label] for label in y[train_indices]]
                indices = random.choices(train_indices, weights, k=len(train_indices))
                train.append(indices)
                test.append(splits[idx])
        return list(zip(train, test))

def hyperParameterTuning(X_train, y_train, objective, cv=None):
    param_tuning = {
        'learning_rate': [0.01, 0.1, 0.30],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': [objective]
    }
    
    cv = StratifiedKFold(n_splits=5) if cv is None else cv
    xgb_model = xgb.XGBClassifier()
    gsearch = GridSearchCV(estimator=xgb_model,
                           param_grid=param_tuning,                        
                           cv=cv,
                           n_jobs=-1,
                           verbose=1)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_

def findBestParams(df, x_columns, y_column, objective, cv=None):
    
    X = np.array(df[x_columns].to_list())
    y = np.array(df[y_column])
    X_train, _, y_train, _ = train_test_split(X, y)
    
    best_params = hyperParameterTuning(X_train, y_train, objective, cv)
    return best_params
    
    
def run_xgboost(df, x_columns, y_column, n_folds=5, splits=None, 
                sampler=False):
    
    X = np.array(df[x_columns].to_list())
    y = np.array(df[y_column])
    
    # First do some hyperparameter tuning
    print('Hyperparameter tuning')
    objective = 'binary:logistic' if len(Counter(y)) == 2 else 'multi:softmax'
    cv = weightedRandomKFold(n_splits=5)
    best_params = findBestParams(df, x_columns, y_column, objective, cv)
    xg_clas = xgb.XGBClassifier(**best_params)
    
    
    true = []
    pred = []
    f1_scores = []
    vm_scores = []
    
    if splits is None:
        cv = StratifiedKFold(n_splits=n_folds)
        splits = cv.split(X, y=y)
    i = 0
    
    print('K-fold cross-validation')
    # K-fold for testing generalizability of the model
    for train_index, test_index, val_index in splits:
        i += 1
        print('Split n° {}'.format(i))
        X_train, X_test, X_val = X[train_index], X[test_index], X[val_index]
        y_train, y_test, y_val = y[train_index], y[test_index], y[val_index]
        
        # if sampler:
        #     smote = SMOTE(random_state=42)
        #     X_train, y_train = smote.fit_resample(X_train, y_train)
        
        if sampler:
            counter = Counter(y_train)
            weights = [len(y_train) / counter[label] for label in y_train]
            train_data = list(zip(X_train, y_train))
            samples = random.choices(train_data, weights, k=len(y_train))
            X_train, y_train = [np.array(i) for i in list(zip(*samples))]
        
        
        xg_clas.fit(X_train,np.array(y_train).reshape(-1), early_stopping_rounds=10,
                    eval_set=[(X_val, y_val)], verbose=False)
        y_pred = xg_clas.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        vm = v_measure_score(y_test, y_pred)
        
        f1_scores.append(f1)
        vm_scores.append(vm)
        
        y_test = [labels[t] for t in y_test]
        y_pred = [labels[p] for p in y_pred]
        
        true.append(y_test)
        pred.append(y_pred)
    
    true = list(itertools.chain.from_iterable(true))
    pred = list(itertools.chain.from_iterable(pred))
    cm = confusion_matrix(true, pred, labels=labels, normalize='true')
    
    return cm, f1_scores, vm_scores, true, pred




# Replace the open function with one that points to the s3 files 
# Tweak read_timeout in case some timeout errors pop up
ENDPOINT_URL = 'https://s3.nautilus.optiputer.net'
            
so_session = Session()
so_config = Config(connect_timeout=50, read_timeout=70)    
s3_client = so_session.client('s3', config=so_config, endpoint_url=ENDPOINT_URL)
   
open = functools.partial(
    open, transport_params={
        'session': so_session,
        'resource_kwargs': {'endpoint_url': ENDPOINT_URL,
                            'config' : so_config}})



parser = argparse.ArgumentParser()
parser.add_argument('--dset', type=str, default='hengen_v1')
parser.add_argument('--load_dataset_from', type=str, 
                    default='/media/louis/DATA1/RNN_NEURO/DATA/DATASETS/hengen_v1.pickle')
parser.add_argument('--label_set', type=str, default='4celltypes')
parser.add_argument('--data_root', type=str,
                    default='s3://hengenlab/RNN_NEURO/DATA/V1_MODEL/HENGENLAB_DIFF_GRATINGS')
parser.add_argument('--stimulus', type=str,
                    default='NM')
parser.add_argument('--disable_comet', default=True)
parser.add_argument('--get_splits_from', type=str,
                    default='./v1_train_test_val_splits')
# parser.add_argument('--get_splits_from', type=str,
#                     default=None)
parser.add_argument('--cutoff', default=False)
parser.add_argument('--sampler', default=True)
parser.add_argument('--fr_binsize', default=200)
parser.add_argument('--fr_nbins', default=50)
parser.add_argument('--isi_nbins', default=100)



args = parser.parse_args()

dset = args.dset
load_dataset_from = args.load_dataset_from
label_set = args.label_set
data_root = args.data_root
stimulus = args.stimulus
disable_comet = args.disable_comet
get_splits_from = args.get_splits_from
cutoff = args.cutoff
sampler = args.sampler
fr_bin_size = args.fr_binsize
fr_nbins = args.fr_nbins
isi_nbins = args.isi_nbins

print('XGBoost {} {}'.format(dset, label_set))

sns.set()

# Plotting parameters
FIGSIZE=(35,30)
SMALL_SIZE=20
MEDIUM_SIZE=25
BIGGER_SIZE=35
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure', figsize=FIGSIZE)


experiment = Experiment(api_key='chA8loyTreVKWEaEe6awZ8xr1',
                                project_name='rnn_neuro',
                                disabled=disable_comet)
experiment.set_name('XGB - {} - {}'.format(dset, label_set))

with open(load_dataset_from, 'rb') as f:
    data_set = pickle.load(f)
    
data_set.apply_len_cutoffs(min_len=30)
if 'v1' in dset:
    keepers = ['e5Rbp4', 'e23Cux2', 'i6Pvalb', 'e4Scnn1a', 'i23Pvalb', 'i23Htr3a',
     'e4Rorb', 'e4other', 'i5Pvalb', 'i4Pvalb', 'i23Sst', 'i4Sst', 'e4Nr5a1',
     'i1Htr3a', 'e5noRbp4', 'i6Sst', 'e6Ntsr1']
    data_set.keep_labels(keepers)
    
data_set.get_labels(dset, label_set, data_root)

if dset == 'neuropixels':
    if label_set == '3celltypes':
        data_set.drop_labels(['other'])

df = data_set.df
if dset == 'calcium':
    # Set an origin
    origin = df['spike_trains'].map(min).min()
    df['spike_trains'] = df['spike_trains'].apply(lambda x: [spike_time - origin for spike_time in x])
df['isis'] = df['spike_trains'].apply(np.diff)

fr_bins = np.arange(start=0, stop=fr_nbins*fr_bin_size, step=fr_bin_size)
firing_rates = []
for sample in df['spike_trains']:
    sample, _ = np.histogram(sample, fr_bins)
    sample = sample / (fr_bin_size / 1000)
    firing_rates.append(sample)
df['firing_rates'] = firing_rates


# With cutoff
if cutoff:
    isi_bins = np.arange(start=0, stop=400, step=2)
    samples = df['isis']

# Without cutoff
else:
    samples = df['isis'].apply(lambda x: np.log(x))
    max_isi = samples.map(max).max()
    min_isi = samples.map(min).min()
    isi_bins = np.linspace(start=min_isi, stop=max_isi, num=isi_nbins)

isi_distribs = []
for sample in samples:
    sample, _ = np.histogram(sample, isi_bins)
    sample = sample / np.sum(sample)
    isi_distribs.append(sample)
df['isi_distribs'] = isi_distribs
df['sorted_isis'] = [np.sort(isi) for isi in df['isis']]


df['fr+isi'] = [np.concatenate((df.iloc[i]['isi_distribs'], df.iloc[i]['firing_rates'])) for i in range(len(df))]


df['labels'] = df['labels'].astype('category')
df['label_names'] = df['labels']
labels = df['labels'].cat.categories
codes = df['labels'].cat.codes
df['labels'] = codes

# df = df.groupby('labels', group_keys=False).apply(lambda x: x.sample(min(len(x), 100)))
df = df.reset_index(drop=True)

# Splits
if get_splits_from is None:
    splits=None
else:
    print('Loading split files...')
    if dset != 'calcium':
        splits = []
        all_splits = os.listdir(get_splits_from)
        all_splits = [split for split in all_splits if label_set in split]
        for split in all_splits:
            split_df = pd.read_csv(os.path.join(get_splits_from, split))
            split_df = split_df[split_df['unit_id'].isin(df['unit_ids'])].reset_index(drop=True)
            split_indices=df['unit_ids'].apply(lambda x: split_df[split_df['unit_id']==x]['set'].values.item())
            train_index = split_indices[split_indices.values == 'train'].index.to_list()
            test_index = split_indices[split_indices.values == 'test'].index.to_list()
            val_index = split_indices[split_indices.values == 'val'].index.to_list()
            splits.append([train_index, test_index, val_index])
    else:
        splits = []
        all_splits = os.listdir(get_splits_from)
        all_splits = [split for split in all_splits if label_set in split]
        for split in all_splits:
            split_df = pd.read_csv(os.path.join(get_splits_from, split))
            train_index = split_df[split_df['set'] == 'train']['index'].to_list()
            test_index = split_df[split_df['set'] == 'test']['index'].to_list()
            val_index = split_df[split_df['set'] == 'val']['index'].to_list()
            splits.append([train_index, test_index, val_index])
            

# Running XGBoost
matrices = []
true_preds = []
all_scores = []
# 1) FR
print('\nFR :')
cm, f1_scores, vm_scores, true, pred = run_xgboost(
    df, 'firing_rates', 'labels', splits=splits, sampler=sampler)
print('F1 score : mean {} - std {}'.format(np.mean(f1_scores), np.var(f1_scores)))
print('V measure : mean {} - std {}'.format(np.mean(vm_scores), np.var(vm_scores)))
true_preds.append([true, pred])
all_scores.append(f1_scores)
matrices.append(cm)

# 2) ISIs
print('\nISI :')
cm, f1_scores, vm_scores, true, pred = run_xgboost(
    df, 'isi_distribs', 'labels',  splits=splits, sampler=sampler)
print('F1 score : mean {} - std {}'.format(np.mean(f1_scores), np.var(f1_scores)))
print('V measure : mean {} - std {}'.format(np.mean(vm_scores), np.var(vm_scores)))
true_preds.append([true, pred])
all_scores.append(f1_scores)
matrices.append(cm)

# 3) FR + ISIs
print('\nFR+ISI :')
cm, f1_scores, vm_scores, true, pred = run_xgboost(
    df, 'fr+isi', 'labels',  splits=splits, sampler=sampler)
print('F1 score : mean {} - std {}'.format(np.mean(f1_scores), np.var(f1_scores)))
print('V measure : mean {} - std {}'.format(np.mean(vm_scores), np.var(vm_scores)))
true_preds.append([true, pred])
all_scores.append(f1_scores)
matrices.append(cm)




fig, axes = plt.subplots(2,2, figsize=(35,30))
features = ['FR distribution', 'ISI distribution', 'FR + ISI distributions']
titles = ['{} by {}'.format(label_set.capitalize(), feature) for feature in features]
for i in range(len(axes.ravel())):
    ax = axes.ravel()[i]
    title = titles[i]
    cm = matrices[i]
    sns.heatmap(cm, vmin=0, vmax=1, cmap='Reds', xticklabels=labels, 
                yticklabels=labels, ax=ax, annot=True, fmt='.3f')
    ax.set_title(title)
    if i == 2:
        break
    
plt.axis('off')
fig.suptitle('{} - {} decoding'.format(dset, label_set))
fig.tight_layout()
plt.subplots_adjust(top=0.85)


experiment.log_figure()
plt.show()
    
experiment.end()



    
