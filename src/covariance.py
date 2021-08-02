import os

import matplotlib.pyplot as plt
import numpy as np
import imageio
import inspect


def compute_cov(x, scaler=None):
    r"""Compute covariance. It can use a custom scaler (fitted to the entire block for example). Or the data is
    z-scored using the average and std computed locally."""
    if scaler is not None:
        if inspect.isclass(scaler):
            scaler = scaler()
            x = scaler.fit_transform(x.T).T
        else:
            x = scaler.transform(x.T).T
    return np.cov(x)


def compute_edge_dist(cov, num_bins=10, vmin=-1, vmax=1.):
    bins = np.linspace(vmin, vmax, num_bins+1)
    return np.apply_along_axis(lambda a: np.histogram(np.clip(a, vmin, vmax), bins=bins, density=True)[0], 1, cov)


def sliding_cov(x, window_size, scaler=None, cov_op=compute_cov):
    r"""Computes a 3d (temporal) covariance matrix over the block using a sliding window.
    ..note ::
        Currently no overlap between windows.
    """
    out = []
    for i in range(0, x.shape[1], window_size):
        x_part = x[:, i:i+window_size]
        cov_part = cov_op(x_part, scaler)
        out.append(cov_part)
    out = np.array(out)
    return out


def create_gif(x, filename):
    r"""Creates a gif out of the 3d matrix :obj:`x` and saves it to :obj:`filename`.
    ..note ::
        This method will generate a lot of images in the directory but will delete them afterwards.
    """
    filenames = []
    for i in range(x.shape[0]):
        plt.imshow(x[i])
        _filename = f'{i}.png'
        filenames.append(_filename)

        # save frame
        plt.savefig(_filename)
        plt.close()

    # build gif
    with imageio.get_writer(filename, mode='I', duration=1) as writer:
        for _filename in filenames:
            image = imageio.imread(_filename)
            writer.append_data(image)

    # Remove files
    for _filename in set(filenames):
        os.remove(_filename)
