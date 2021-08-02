from torch.nn import AvgPool1d
from torch import tensor
import numpy as np


def add_random_noise(x, sigma=0.1):
    noisy_x = []
    for hist in x:
        hist = list(hist)
        noisy_hist = np.asarray([xi*np.random.normal(1,sigma) for xi in hist])
        noisy_x.append(noisy_hist)
    return noisy_x


def moving_average(x, kernel_width=3,padding=1):
    averaged_x = []
    m = AvgPool1d(kernel_width,1,padding,count_include_pad=False)
    for hist in x:
        hist = list(hist)
        averaged_hist = m(tensor([[hist]])).numpy()[0,0,:]
        averaged_x.append(averaged_hist)
    return averaged_x


def sliding_windows(window_size, max_time, step):
    ''' Creates overlapping windows '''
    return (np.expand_dims([0,window_size], 0) \
                + np.expand_dims(np.arange(max_time, step=step), 0).T)


def slice_data(X, y, window_size):
    ''' Slices spike trains into chunks whose size is a percentage (slice_perc) of the
    sample's length '''

    new_X = []
    new_y = []
    for sample, label in list(zip(X, y)):
        augmented_sample = []
        max_time = 3 # 3s = duration of a trial
        step_size = window_size
        windows = sliding_windows(window_size, max_time, step_size)
        for window in windows:
            start, end = window
            augmented_sample.append([time for time in sample if start<time<end])
        new_X.append(augmented_sample)
        new_y.append(label)
    return new_X, new_y


def slice_data_old(X, y, slice_perc):
    ''' Slices spike trains into chunks whose size is a percentage (slice_perc) of the
    sample's length '''

    augmented_X = []
    augmented_y = []
    for sample, label in list(zip(X, y)):
        window_size = int(slice_perc * len(sample))
        max_size = len(sample)
        step_size = window_size
        windows = sliding_windows(window_size, max_size, step_size)
        for window in windows:
            start, end = window
            augmented_X.append(sample[start:end])
            augmented_y.append(label)
    return augmented_X, augmented_y
