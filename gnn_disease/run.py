import argparse

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Data

from gnn_disease.utils import *
from gnn_disease.model import *
from gnn_disease.model_g_encoder import *
from gnn_disease.model_kg_encoder import *
from gnn_disease.trainer import Trainer


parser = argparse.ArgumentParser(description="PyTorch JCLModel")
parser.add_argument("--data", default="./data", help="path to dataset")
parser.add_argument("--h_dim", default=32, type=int, help="dimension of layer h")
parser.add_argument("--z_dim", default=32, type=int, help="dimension of layer z")
parser.add_argument("--tau", default=1.0, type=float, help="softmax temperature")
parser.add_argument("--lr", default=0.003, type=float, help="learning rate")
parser.add_argument("--epochs", default=100, type=int, help="train epochs")
parser.add_argument(
    "--disable-cuda", default=True, action="store_true", help="disable CUDA"
)
parser.add_argument(
    "--log-every-n-steps", default=1, type=int, help="log every n steps"
)
parser.add_argument("--g_name", default="GCN", type=str, help="log every n steps")
parser.add_argument("--kg_name", default="RGCN", type=str, help="log every n steps")

args = parser.parse_args()

device = torch.device(
    "cuda" if not args.disable_cuda and torch.cuda.is_available() else "cpu"
)

# Load knowledge graph statistics
with open(args.data + "/entity2id.txt", "r") as f:
    num_ents = (int)(f.readline())
with open(args.data + "/relation2id.txt", "r") as f:
    num_rels = (int)(f.readline())

# Load GO knowledge graph for RGCN Model
train_triples = load_triples(args.data)
edge_index, edge_type = get_kg_data(train_triples, num_rels)
kg_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_ents)

# Load human net for GCN Model
hnadj = load_sparse(args.data + "/hnet.npz")
src = hnadj.row
dst = hnadj.col
hn_edge_weight = torch.tensor(np.hstack((hnadj.data, hnadj.data)), dtype=torch.float)
hn_edge_weight = (hn_edge_weight - hn_edge_weight.min()) / (
    hn_edge_weight.max() - hn_edge_weight.min()
)
hn_edge_index = torch.tensor(
    np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))),
    dtype=torch.long,
)

# Load gene2GO align
g2o = load_sparse(args.data + "/g2o.npz")
g2o = mx_to_torch_sparse_tesnsor(g2o).to_dense()

x = generate_sparse_one_hot(g2o.shape[0])
g_data = Data(x=x, edge_index=hn_edge_index, edge_weight=hn_edge_weight)

d2g = load_sparse(args.data + "/d2g.npz")
d2g = mx_to_torch_sparse_tesnsor(d2g)
if args.g_name == "GCN":
    g_encoder = GCN(nfeat=g_data.x.shape[1], nhid=args.h_dim)
elif args.g_name == "EG":
    g_encoder = EG(nfeat=g_data.x.shape[1], nhid=args.h_dim)
elif args.g_name == "GAT":
    g_encoder = GAT(nfeat=g_data.x.shape[1], nhid=args.h_dim)
elif args.g_name == "GATv2":
    g_encoder = GATv2(nfeat=g_data.x.shape[1], nhid=args.h_dim)
elif args.g_name == "GraphGEN":
    g_encoder = GraphGEN(nfeat=g_data.x.shape[1], nhid=args.h_dim)
else:
    raise NotImplementedError
if args.kg_name == "FastRGCN":
    kg_encoder = FastRGCN(num_nodes=num_ents, nhid=args.h_dim, num_rels=num_rels * 2)
elif args.kg_name == "RGATConvNet":
    kg_encoder = RGATConvNet(num_nodes=num_ents, nhid=args.h_dim, num_rels=num_rels * 2)
elif args.kg_name == "RGCN":
    kg_encoder = RGCN(num_nodes=num_ents, nhid=args.h_dim, num_rels=num_rels * 2)

projection = Projection(args.h_dim, args.z_dim)
model = GNNDisease(g_encoder, kg_encoder, projection)
opt = optim.RMSprop(model.parameters(), args.lr)
trainer = Trainer(
    model,
    tau=args.tau,
    optimizer=opt,
    log_every_n_steps=args.log_every_n_steps,
    device=device,
    name=args.g_name + "-" + args.kg_name,
)
trainer.load_data(g_data, kg_data, g2o, d2g, args.data)
print("Finish initializing...")
print("---------------------------------------")
trainer.train(args.epochs)


def test(path):
    checkpoint = torch.load(path)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    auroc, ap, a, tk = trainer.infer()
