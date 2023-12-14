import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv, RGATConv, RGCNConv


class RGCN(nn.Module):
    def __init__(self, num_nodes, nhid, num_rels, num_bases=None):
        super(RGCN, self).__init__()
        self.rconv1 = RGCNConv(
            num_nodes, nhid, num_rels, bias=True, num_bases=num_bases
        )
        self.rconv2 = RGCNConv(nhid, nhid, num_rels, bias=True, num_bases=num_bases)

    def forward(self, data):
        x = self.rconv1(None, data.edge_index, data.edge_type)
        x = F.leaky_relu(x)
        x = self.rconv2(x, data.edge_index, data.edge_type)
        return x


class FastRGCN(nn.Module):
    def __init__(self, num_nodes, nhid, num_rels, num_bases=None):
        super(FastRGCN, self).__init__()
        self.rconv1 = FastRGCNConv(
            num_nodes, nhid, num_rels, bias=True, num_bases=num_bases
        )
        self.rconv2 = FastRGCNConv(nhid, nhid, num_rels, bias=True, num_bases=num_bases)

    def forward(self, data):
        x = self.rconv1(None, data.edge_index, data.edge_type)
        x = F.leaky_relu(x)
        x = self.rconv2(x, data.edge_index, data.edge_type)

        return x


class RGATConvNet(nn.Module):
    def __init__(self, num_nodes, nhid, num_rels, num_bases=None):
        super(RGATConvNet, self).__init__()
        self.rconv1 = RGCNConv(
            num_nodes, nhid, num_rels, bias=True, num_bases=num_bases
        )
        self.rconv2 = RGATConv(nhid, nhid, num_rels, num_bases=None)

    def forward(self, data):

        x = self.rconv1(None, data.edge_index, data.edge_type)
        x = F.leaky_relu(x)
        x = self.rconv2(x, data.edge_index, data.edge_type)
        return x
