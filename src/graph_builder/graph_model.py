# TODO: implement GCN, GAT and SAGE
from torch import nn
from torch.nn import Module, ModuleList
from dgl.nn.pytorch import GraphConv, SAGEConv, EdgeWeightNorm


class WGCN(Module):
    def __init__(self, n_infeat, n_hidden, n_classes, n_layers, activation):
        super(WGCN, self).__init__()
        self.layers = ModuleList()
        self.layers.append(
            GraphConv(n_infeat, n_hidden, weight=True, activation=activation)
        )
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, weight=True, activation=activation)
            )
        self.layers.append(
            GraphConv(
                n_hidden,
                n_classes,
                weight=True,
            )
        )
        # TODO: change it norm='right' in case of zero or non positive values else 'both'
        self.edge_norm = EdgeWeightNorm(norm="right")
        self.softmax = nn.Softmax()

    def forward(self, g, features, edge_weight):
        h = features

        norm_edge_weight = self.edge_norm(g, abs(edge_weight))
        for i, layer in enumerate(self.layers):
            # handle api changes for differnt DGL version
            h = layer(g, h, edge_weight=norm_edge_weight)
        return self.softmax(h)
