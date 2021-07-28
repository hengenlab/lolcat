import numpy as np

def compute_fr_dist(x, bins=200):
    r"""Compute FR Histogram over a full range of values."""
    if type(bins) == int:
        num_bins = bins
        x_dist = np.empty((x.shape[0],num_bins))
    else:
        num_bins = len(bins)-1
        x_dist = np.empty((x.shape[0],num_bins))
    
    for i, xi in enumerate(x):
                       
        x_dist[i,:] = np.histogram(xi,bins=int(num_bins))[0]
    return x_dist