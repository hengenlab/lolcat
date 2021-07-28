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
        