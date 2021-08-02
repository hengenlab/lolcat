import os
import re
import pickle
from collections import OrderedDict
from functools import wraps

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn import AvgPool1d
from torch import tensor
import numpy as np


def add_random_noise(x, augmentation_perc=1, sigma=0.1):
    r"""Add/subtract a random amount of gaussian noise (sigma=std. dev.) to a subset (augmentation_perc=prob. to affect a bin) of histogram bins"""
    add_noise = lambda hist: np.multiply(hist, np.random.normal(1, sigma, hist.size))
    noisy_x = np.apply_along_axis(add_noise, 1, x)
    draws = np.where(np.random.uniform(0, 1, noisy_x.shape[1]) > augmentation_perc)[0]
    noisy_x[draws, :] = x[draws, :]
    return noisy_x


def moving_average(x, augmentation_perc=1, kernel_width=3):
    r"""Rolling (kernel_width is diameter not radius) average of a subset (augmentation_perc=prob. to affect a bin) of histogram bins"""
    assert kernel_width % 2 == 1
    padding = kernel_width // 2
    average = lambda hist: np.convolve(hist, np.ones(kernel_width) / kernel_width, mode='same')
    averaged_x = np.apply_along_axis(average, 1, x)
    draws = np.where(np.random.uniform(0, 1, averaged_x.shape[1]) > augmentation_perc)[0]
    averaged_x[draws, :] = x[draws, :]
    return averaged_x

# @LOUIS CODE AUGMENTATIONS TO SPIKES HERE
def crop_spike_train(spike_train, crop_perc):
    window_size = int(crop_perc * len(spike_train))
    start = np.random.uniform(0, window_size - 1)
    stop = start + window_size
    return spike_train[start, stop]


def pad_jagged(matrix):
    ''' Pad a jagged matrix '''
    maxlen = max(len(row) for row in matrix)

    padded_matrix = np.zeros((len(matrix), maxlen))
    for i, row in enumerate(matrix):
        padded_matrix[i, :len(row)] += row
    return padded_matrix


def crop_data(x, augmentation_perc=1, crop_perc=0.5):
    r'''Crop a subset (augmentation_perc=prob. to affect a spike train) of spike trains, taking a window of size crop_perc * len(spike_train)'''
    cropped_x = np.apply_along_axis(crop_spike_train, 1, x, crop_perc=crop_perc)
    draws = np.where(np.random.uniform(0, 1, cropped_x.shape[1]) > augmentation_perc)[0]
    cropped_x[draws, :] = x[draws, :]
    return cropped_x



class FiringRates:
    def __init__(self, window_size, bin_size):
        self.window_size = window_size
        self.bin_size = bin_size

        self.num_bins = int(window_size / self.bin_size)
        self.bins = np.linspace(0, self.window_size, self.num_bins + 1)

    def __call__(self, X):
        X_binned = np.zeros((X.shape[0], self.num_bins))
        for i, x in enumerate(X):
            X_binned[i] = np.histogram(x, self.bins)[0].astype(int)
        return X_binned


class ISIDistribution:
    def __init__(self, bins, min_isi=0, max_isi=0.4, log=False, adaptive=False, window_size=None):
        self.bins = bins
        if isinstance(bins, int):
            self.num_bins = bins
        else:
            self.num_bins = len(bins) - 1
        self.min_isi = min_isi
        self.max_isi = max_isi

        self.log = log
        self.adaptive = adaptive
        self.augmentation_percs = augmentation_percs
        self.preaugmentation_percs = preaugmentation_percs
        self.window_size = window_size

    def __call__(self, X):
        if self.window_size:
            num_windows = int(3 / self.window_size)
            X_isi = np.zeros((len(X), num_windows, self.num_bins - 1))  # get rid of the 0 bin
            for i, x in enumerate(X):
                x = pad_jagged(x)
                # compute isi
                x = np.diff(x)
                # compute histogram for each mini window
                x = np.clip(x, a_min=self.min_isi, a_max=self.max_isi)
                if self.log == True:
                    x = np.log10(x)
                X_isi[i] = np.apply_along_axis(lambda m: np.histogram(m, self.bins)[0][1:].astype(int), 1, x)
        else:
            X_isi = np.zeros((X.shape[0], self.num_bins))
            if self.adaptive == True:
                X = [np.clip(np.diff(x), a_min=self.min_isi, a_max=self.max_isi) for x in X]
                min_X = min([np.min(x) for x in X if len(x) > 0])
                max_X = max([np.max(x) for x in X if len(x) > 0])
                X_vals = np.hstack(X)
                percs = np.linspace(0, 1, self.num_bins + 1)
                adaptive_bins = np.percentile(X_vals, percs)
                self.bins = adaptive_bins
                for i, x in enumerate(X):
                    X_isi[i] = np.histogram(x, self.bins)[0].astype(int)
            else:
                for i, x in enumerate(X):
                    # compute isi
                    x = np.diff(x)
                    # compute histogram
                    x = np.clip(x, a_min=self.min_isi, a_max=self.max_isi)
                    if self.log == True:
                        x = np.log10(x)
                    X_isi[i] = np.histogram(x, self.bins)[0].astype(int)
            augmentation_percs = self.augmentation_percs
            if augmentation_percs[0] > 0:
                X_isi = add_random_noise(X_isi, augmentation_percs[0])
            if augmentation_percs[1] > 0:
                X_isi = moving_average(X_isi, augmentation_percs[1])
        return X_isi


def compute_hist(X):
    X_isi = np.zeros((X.shape[0], 100))
    for i, x in enumerate(X):
        # compute isi
        x = np.diff(x)
        # compute histogram
        x = np.clip(x, a_min=0., a_max=0.4)
        X_isi[i] = np.histogram(x, 100)[0].astype(int)
    return X_isi

class ConcatFeats:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, X):
        X_out = []
        for transform in self.transforms:
            X_out.append(transform(X))
        return np.column_stack(X_out)
