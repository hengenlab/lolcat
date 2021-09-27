import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add
from torch_geometric.utils import softmax



class MLP(nn.Module):
    r"""Multi-layer perceptron model, with optional batchnorm layers.

    Args:
        hidden_layers (list): List of layer dimensions, from input layer to output layer. If first input size is -1,
            will use a lazy layer.
        bias (boolean, optional): If set to :obj:`True`, bias will be used in linear layers. (default: :obj:`True`).
        activation (torch.nn.Module, optional): Activation function. (default: :obj:`nn.ReLU`).
        batchnorm (boolean, optional): If set to :obj:`True`, batchnorm layers are added after each linear layer, before
            the activation (default: :obj:`False`).
        drop_last_nonlin (boolean, optional): If set to :obj:`True`, the last layer won't have activations or
            batchnorm layers. (default: :obj:`True`)

    Examples:
        >>> m = MLP([-1, 16, 64])
        MLP(
          (layers): Sequential(
            (0): LazyLinear(in_features=0, out_features=16, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=64, bias=True)
          )
        )
    """
    def __init__(self, hidden_layers, *, bias=True, activation=nn.ReLU(True), batchnorm=False, drop_last_nonlin=True, dropout=0.):
        super().__init__()

        # build the layers
        layers = []
        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            if in_dim == -1:
                layers.append(nn.LazyLinear(out_dim, bias=bias))
            else:
                layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            if batchnorm:
                layers.append(nn.BatchNorm1d(num_features=out_dim))
            if activation is not None:
                layers.append(activation)
            if dropout > 0.:
                layers.append(nn.Dropout(dropout))

        # remove activation and/or batchnorm layers from the last block
        if drop_last_nonlin:
            remove_layers = -(int(activation is not None) + int(batchnorm) + int(dropout>0.))
            if remove_layers:
                layers = layers[:remove_layers]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class GlobalPoolingModel(nn.Module):
    def __init__(self, encoder, classifier, pool=global_mean_pool):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.pool = pool

    def forward(self, x, batch, return_trial_embeddings=False, return_attention=False):
        emb = self.encoder(x)  # all trial sequences are encoded

        # compute global cell-wise embedding
        if isinstance(self.pool, GlobalAttention) and return_attention:
            global_emb, gate = self.pool(emb, batch, return_attention=True)
        else:
            global_emb = self.pool(emb, batch)

        # classify
        out = self.classifier(global_emb)
        logits = torch.log_softmax(out, dim=1)

        if not return_trial_embeddings:
            return logits
        elif return_attention:
            return logits, emb, global_emb, gate
        else:
            return logits, emb, global_emb


class GlobalAttention(torch.nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(self, in_channels, out_channels, heads=1):
        super(GlobalAttention, self).__init__()

        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels, heads, bias=False),
        )
        self.nn = MLP([in_channels, out_channels, out_channels])

    def forward(self, x, batch, return_attention=False):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        if not return_attention:
            return out
        else:
            return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)


class MultiHeadPooling(torch.nn.Module):
    def __init__(self, *pool):
        super(MultiHeadPooling, self).__init__()

        self.pool = nn.ModuleList(pool)

    def forward(self, x, batch):
        out = []
        for pool in self.pool:
            out.append(pool(x, batch))
        out = torch.cat(out, dim=1)
        return out



class GlobalMultiHeadAttention(torch.nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(self, in_channels, out_channels, heads=1):
        super(GlobalMultiHeadAttention, self).__init__()

        self.heads = heads
        self.out_channels = out_channels

        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels, heads, bias=False),
        )
        self.nn = MLP([in_channels, out_channels, out_channels])

    def forward(self, x, batch, return_attention=False):
        """"""
        size = batch[-1].item() + 1

        # compute attention
        alpha = self.gate_nn(x).view(-1, 1, self.heads)

        x = self.nn(x).view(-1, self.out_channels, self.heads)



        gate = softmax(gate, batch, num_nodes=size, dim=2)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        if not return_attention:
            return out
        else:
            return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)
