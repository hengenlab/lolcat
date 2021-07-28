from collections import OrderedDict

import torch.nn as nn


class MLP(nn.Module):
    r"""Multi-layer perceptron.

    Args:
        input_dims (int): Input dimension.
        n_hiddens (int or list): Hidden layer dimensions.
        n_class (int): Number of classes.
    """
    def __init__(self, input_dims, n_hiddens, n_class, dropout_p=0.2):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        self.n_hiddens = n_hiddens
        self.n_class = n_class
        self.dropout_p = dropout_p

        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            if dropout_p > 0.:
                layers['drop{}'.format(i+1)] = nn.Dropout(dropout_p)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model= nn.Sequential(layers)

    def forward(self, input):
        r"""Inputs can be 2d images"""
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        output = self.model.forward(input)
        return nn.functional.log_softmax(output, dim=1)

    def hook(model, input, output):
        return output.detach()

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def register_hooks(self):
        self.activation = {}
        hooks = {}
        for name, module in self.named_modules():
            hooks[name] = module.register_forward_hook(self.get_activation(name))
        return hooks
