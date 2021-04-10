import numpy as np

def compute_isi_dist(x, bins=200, min_isi=0, max_isi=0.4):
    r"""Compute ISI Histogram over a range of values (specify min_isi,max_isi,& bin_size in ms)."""
    if type(bins) == int:
        num_bins = bins
        x_dist = np.empty((x.shape[0],num_bins))
    else:
        num_bins = len(bins)-1
        x_dist = np.empty((x.shape[0],num_bins))
    
    for i, xi in enumerate(x):
        if type(bins) == int:
            xi = xi[(xi>=min_isi)&(xi<=max_isi)]
                       
        x_dist[i,:] = np.histogram(xi,bins=int(num_bins))[0]
    return x_dist