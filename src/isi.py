import numpy as np

def compute_isi_dist(x, bins=200, min_isi=0, max_isi=0.4):
    r"""Compute ISI Histogram over a range of values (specify min_isi,max_isi,& bin_size in ms)."""
    f,p=0,0
    if type(bins) == int:
        num_bins = bins
        x_dist = np.empty((x.shape[0],num_bins))
    else:
        num_bins = len(bins)-1
        #x_dist = np.empty((x.shape[0],num_bins))
        x_dist = np.empty((x.shape[0],num_bins))
    
    for i, xi in enumerate(x):
        xi = xi[(xi>=min_isi)&(xi<=max_isi)]
        if type(bins) == int:
            x_dist[i,:] = np.histogram(xi,bins=int(num_bins))[0]
        else:
            x_dist[i,:] = np.histogram(xi,bins=bins)[0]
            p+=1
            
    return x_dist