import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EGConv, GATConv, GATv2Conv, GCNConv, GENConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid, bias=True)
        self.conv2 = GCNConv(nhid, nhid, bias=True)

    def forward(self, data):
        try:
            x = self.conv1(data.x, data.edge_index, data.edge_weight)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index, data.edge_weight)
        except:
            x = self.conv1(data.x, edge_index=data.edge_index)
            x = F.leaky_relu(x)
            x = self.conv2(x, edge_index=data.edge_index)
        return x


class EG(nn.Module):
    def __init__(self, nfeat, nhid):
        super(EG, self).__init__()
        self.conv1 = EGConv(nfeat, nhid, bias=True)
        self.conv2 = EGConv(nhid, nhid, bias=True)

    def forward(self, data):
        try:
            x = self.conv1(data.x, data.edge_index, data.edge_weight)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index, data.edge_weight)
        except:
            x = self.conv1(data.x, edge_index=data.edge_index)
            x = F.leaky_relu(x)
            x = self.conv2(x, edge_index=data.edge_index)
        return x


class GATv2(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(nfeat, nhid, bias=True)
        self.conv2 = GATv2Conv(nhid, nhid, bias=True)

    def forward(self, data):
        try:
            x = self.conv1(data.x, data.edge_index, data.edge_weight)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index, data.edge_weight)
        except:
            x = self.conv1(data.x, data.edge_index)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GAT, self).__init__()
        self.conv1 = GATConv(nfeat, nhid, bias=True)
        self.conv2 = GATConv(nhid, nhid, bias=True)

    def forward(self, data):
        try:
            x = self.conv1(data.x, data.edge_index, data.edge_weight)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index, data.edge_weight)
        except:
            x = self.conv1(data.x, data.edge_index)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index)
        return x


class GraphGEN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GraphGEN, self).__init__()
        self.conv1 = GENConv(nfeat, nhid, bias=True)
        self.conv2 = GENConv(nhid, nhid, bias=True)

    def forward(self, data):
        try:
            x = self.conv1(data.x, data.edge_index, data.edge_weight)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index, data.edge_weight)
        except:
            x = self.conv1(data.x, data.edge_index)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index)
        return x
