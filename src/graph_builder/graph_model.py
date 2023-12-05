# TODO: implement GCN, GAT and SAGE
import torch
from torch import nn
from torch.nn import Module, ModuleList
from dgl.nn.pytorch import GraphConv, SAGEConv, EdgeWeightNorm, GINEConv, EGATConv
from torch.nn.functional import relu, leaky_relu, softmax, elu, log_softmax


class WGCN(Module):
    def __init__(self, n_infeat, n_hidden, n_classes, n_layers, activation):
        super(WGCN, self).__init__()
        self.layers = ModuleList()
        self.layers.append(
            GraphConv(n_infeat, n_hidden, weight=True, activation=activation) + nn.Parameter(torch.FloatTensor(n_hidden))
        )
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, weight=True, activation=activation) + nn.Parameter(torch.FloatTensor(n_hidden))
            )
        self.layers.append(
            GraphConv(
                n_hidden,
                n_classes,
                weight=True,
            ) + nn.Parameter(torch.FloatTensor(n_classes))
        )
        # TODO: change it norm='right' in case of zero or non positive values else 'both'
        self.edge_norm = EdgeWeightNorm(norm="right")
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g, features, edge_weight):
        h = features

        # norm_edge_weight = self.edge_norm(g, abs(edge_weight))
        norm_edge_weight = edge_weight
        for i, layer in enumerate(self.layers):
            # handle api changes for differnt DGL version
            h = self.dropout(h)
            h = layer(g, h, edge_weight=norm_edge_weight)
        return log_softmax(h, dim=1)


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = relu(h)
        h = self.conv2(g, h)
        return h


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = softmax(nodes.mailbox["e"], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata["z"] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop("h")


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == "cat":
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = elu(h)
        h = self.layer2(h)
        return h


class GINE(Module):
    """
    Graph Isomorphism Network with Edge Features, introduced by Strategies for Pre-training Graph Neural Networks
    check the implementation
    https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.GINEConv.html#dgl.nn.pytorch.conv.GINEConv

    """

    def __init__(self, n_infeat, n_hidden, n_classes, n_layers, activation):
        super(GINE, self).__init__()
        self.layers = ModuleList()
        self.layers.append(
            GINEConv(n_infeat, n_hidden, weight=True, activation=activation)
        )
        for i in range(n_layers - 1):
            self.layers.append(
                GINEConv(n_hidden, n_hidden, weight=True, activation=activation)
            )
        self.layers.append(
            GINEConv(
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

        # norm_edge_weight = self.edge_norm(g, abs(edge_weight))
        norm_edge_weight = edge_weight
        for i, layer in enumerate(self.layers):
            # handle api changes for differnt DGL version
            h = layer(g, h, edge_weight=norm_edge_weight)
        return self.softmax(h)


class EGAT(Module):
    """
    Graph attention layer that handles edge features from Rossmann-Toolbox (see supplementary data)

    The difference lies in how unnormalized attention scores eij
     are obtained:    check the implementation
    https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.EGATConv.html#dgl.nn.pytorch.conv.EGATConv
    """

    def __init__(self, n_infeat, n_hidden, n_classes, n_layers, activation):
        super(EGAT, self).__init__()
        self.layers = ModuleList()
        self.layers.append(
            EGATConv(n_infeat, n_hidden, weight=True, activation=activation)
        )
        for i in range(n_layers - 1):
            self.layers.append(
                EGATConv(n_hidden, n_hidden, weight=True, activation=activation)
            )
        self.layers.append(
            EGATConv(
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

        # norm_edge_weight = self.edge_norm(g, abs(edge_weight))
        norm_edge_weight = edge_weight
        for i, layer in enumerate(self.layers):
            # handle api changes for differnt DGL version
            h = layer(g, h, edge_weight=norm_edge_weight)
        return self.softmax(h)
