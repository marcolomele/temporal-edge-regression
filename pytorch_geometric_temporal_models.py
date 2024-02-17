import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

### MPNN LSTM ###
class MPNNLSTM(nn.Module):
    r"""An implementation of the Message Passing Neural Network with Long Short Term Memory.
    For details see this paper: `"Transfer Graph Neural Networks for Pandemic Forecasting." <https://arxiv.org/abs/2009.08388>`_

    Args:
        in_channels (int): Number of input features.
        hidden_size (int): Dimension of hidden representations.
        num_nodes (int): Number of nodes in the network.
        window (int): Number of past samples included in the input.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_nodes: int,
        window: int,
        dropout: float,
    ):
        super(MPNNLSTM, self).__init__()

        self.window = window
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.in_channels = in_channels

        self._create_parameters_and_layers()

    def _create_parameters_and_layers(self):

        self._convolution_1 = GCNConv(self.in_channels, self.hidden_size)
        self._convolution_2 = GCNConv(self.hidden_size, self.hidden_size)

        self._batch_norm_1 = nn.BatchNorm1d(self.hidden_size)
        self._batch_norm_2 = nn.BatchNorm1d(self.hidden_size)

        self._recurrent_1 = nn.LSTM(2 * self.hidden_size, self.hidden_size, 1)
        self._recurrent_2 = nn.LSTM(self.hidden_size, self.hidden_size, 1)

    def _graph_convolution_1(self, X, edge_index, edge_weight):
        X = F.relu(self._convolution_1(X, edge_index, edge_weight))
        X = self._batch_norm_1(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        return X

    def _graph_convolution_2(self, X, edge_index, edge_weight):
        X = F.relu(self._convolution_2(X, edge_index, edge_weight))
        X = self._batch_norm_2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        return X

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Making a forward pass through the whole architecture.

        Arg types:
            * **X** *(PyTorch FloatTensor)* - Node features.
            * **edge_index** *(PyTorch LongTensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch LongTensor, optional)* - Edge weight vector.

        Return types:
            *  **H** *(PyTorch FloatTensor)* - The hidden representation of size 2*nhid+in_channels+window-1 for each node.
        """
        R = list()

        S = X.view(-1, self.window, self.num_nodes, self.in_channels)
        S = torch.transpose(S, 1, 2)
        S = S.reshape(-1, self.window, self.in_channels)
        O = [S[:, 0, :]]

        for l in range(1, self.window):
            O.append(S[:, l, self.in_channels - 1].unsqueeze(1))

        S = torch.cat(O, dim=1)

        X = self._graph_convolution_1(X, edge_index, edge_weight)
        R.append(X)

        X = self._graph_convolution_2(X, edge_index, edge_weight)
        R.append(X)

        X = torch.cat(R, dim=1)

        X = X.view(-1, self.window, self.num_nodes, X.size(1))
        X = torch.transpose(X, 0, 1)
        X = X.contiguous().view(self.window, -1, X.size(3))

        X, (H_1, _) = self._recurrent_1(X)
        X, (H_2, _) = self._recurrent_2(X)

        H = torch.cat([H_1[0, :, :], H_2[0, :, :], S], dim=1)
        return H
    
### EVOLVEGCN-H ###
    
from torch.nn import GRU
from torch_geometric.nn import TopKPooling
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GCNConv_Fixed_W(MessagePassing):
    r"""The graph convolutional operator adapted from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, with weights not trainable.
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.
    Its node-wise formulation is given by:
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j
    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)
    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv_Fixed_W, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, W, x, edge_index, edge_weight):

        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)

        x = torch.matmul(x, W)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out


    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

class EvolveGCNH(torch.nn.Module):
    r"""An implementation of the Evolving Graph Convolutional Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_

    Args:
        num_of_nodes (int): Number of vertices.
        in_channels (int): Number of filters.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
    """

    def __init__(
        self,
        num_of_nodes: int,
        in_channels: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
    ):
        super(EvolveGCNH, self).__init__()

        self.num_of_nodes = num_of_nodes
        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.weight = None
        self.initial_weight = torch.nn.Parameter(torch.Tensor(in_channels, in_channels))
        self._create_layers()
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.initial_weight)


    def _create_layers(self):

        self.ratio = self.in_channels / self.num_of_nodes

        self.pooling_layer = TopKPooling(self.in_channels, self.ratio)

        self.recurrent_layer = GRU(
            input_size=self.in_channels, hidden_size=self.in_channels, num_layers=1
        )

        self.conv_layer = GCNConv_Fixed_W(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            improved=self.improved,
            cached=self.cached,
            normalize=self.normalize,
            add_self_loops=self.add_self_loops
        )

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.

        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        X_tilde = self.pooling_layer(X, edge_index)
        X_tilde = X_tilde[0][None, :, :]
        if self.weight is None:
            self.weight = self.initial_weight.data
        W = self.weight[None, :, :]
        X_tilde, W = self.recurrent_layer(X_tilde, W)
        X = self.conv_layer(W.squeeze(dim=0), X, edge_index, edge_weight)
        return X