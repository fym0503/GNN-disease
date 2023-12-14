import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--plot_auprc", type=bool, default=True)
parser.add_argument("--plot_auroc", type=bool, default=True)
parser.add_argument("--plot_loss", type=bool, default=True)

args = parser.parse_args()
g_names = ["GCN", "GAT", "GATv2", "GraphGEN", "EG"]
kg_names = ["RGCN", "RGATConvNet"]
os.makedirs(args.output_dir, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 8))

for i in range(len(g_names)):
    for j in range(len(kg_names)):
        with open(
            args.input_dir + "/" + g_names[i] + "-" + kg_names[j] + "/training.log"
        ) as f:
            AUROC_all = []
            epoch = []
            count = 0
            for line in f:
                if "DEBUG" in line:
                    item = line.split("\t")
                    AUROC = float(item[2].split(" ")[1])
                    AUROC_all.append(AUROC)
                    epoch.append(count)
                    count = count + 1
            sns.lineplot(
                x=np.array(epoch),
                y=np.array(AUROC_all),
                label=g_names[i] + "-" + kg_names[j],
            )
plt.xlabel("Epoch")
plt.ylabel("AUROC")
plt.savefig(args.output_dir + "/AUROC.png", dpi=600)
plt.cla()

for i in range(len(g_names)):
    for j in range(len(kg_names)):
        with open(
            args.input_dir + "/" + g_names[i] + "-" + kg_names[j] + "/training.log"
        ) as f:
            AP_all = []
            epoch = []
            count = 0
            for line in f:
                if "DEBUG" in line:
                    item = line.split("\t")
                    AP = float(item[3].split(" ")[1])
                    AP_all.append(AP)
                    epoch.append(count)
                    count = count + 1
            sns.lineplot(
                x=np.array(epoch),
                y=np.array(AP_all),
                label=g_names[i] + "-" + kg_names[j],
            )
plt.xlabel("Epoch")
plt.ylabel("AUPRC")
plt.savefig(args.output_dir + "/AUPRC.png", dpi=600)
plt.cla()

for i in range(len(g_names)):
    for j in range(len(kg_names)):
        with open(
            args.input_dir + "/" + g_names[i] + "-" + kg_names[j] + "/training.log"
        ) as f:
            loss_all = []
            epoch = []
            count = 0
            for line in f:
                if "DEBUG" in line:
                    item = line.split("\t")
                    loss = float(item[1].split(" ")[1])
                    loss_all.append(loss)
                    epoch.append(count)
                    count = count + 1
            sns.lineplot(
                x=np.array(epoch),
                y=np.array(loss_all),
                label=g_names[i] + "-" + kg_names[j],
            )
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(args.output_dir + "/Loss.png", dpi=600)
plt.cla()
