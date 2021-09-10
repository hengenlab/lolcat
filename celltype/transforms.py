import copy

import torch


class Dropout:
    def __init__(self, dropout_p):
        self.dropout_p = dropout_p

    def __call__(self, data):
        # make a copy from the data object
        data = copy.deepcopy(data)

        # dropout entire trials
        x = data.x
        dropout_mask = torch.empty((x.size(0),), dtype=torch.float32, device=x.device).uniform_(0, 1) > self.dropout_p
        data.x = x[dropout_mask]
        return data
