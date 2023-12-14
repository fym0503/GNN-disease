# GNN-Disease: Learning Disease Similarity from Gene Graph Databases
This is the course project repo for CSCI-5120 Course in The Chinese University of Hong Kong. The project title is: GNN-Disease: Learning Disease Similarity from Gene Graph Databases

**Group Name**: AIH

**Group Member**: Yixuan Wang, Yimin Fan, Liang Hong, Jiyue Jiang

**Project Description**: 
In this project, Our goal is to quantify the similarity between disease using existing graph databases. First, we collect data from two common human gene databases called HumanNet and Go Database. Then we use Graph neural network to embed th gene graph and perform contrastive learning in the gene embedding space to obtain high-quality gene embeddings. In the inference stage, we first collect the genes related to specific disease, and use average pooling to get the disease embedding and analyze the similarity across disease through the similarity of disease embeddings

## Environment Setup
```bash
conda create -n gnn_disease python=3.9
pip install -r requirements.txt
```
## Usage
1. Dataset Preparation
   Download the dataset from this url and unzip it, put it into the data folder
   ```bash
   https://drive.google.com/file/d/1vnWfWbR_lb6uZ-mC0m7RXZ8dPlB-o5kR/view?usp=sharing
   ```
2. Model Training
   Our model supports multiple graph neural network. The edges in the HumanNet graph database do not contain much information, so we can use the below networks for encoding
   ```bash
   GCN https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv
   GAT https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html#torch_geometric.nn.conv.GATConv
   GATv2 https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv
   EG https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.EGConv.html#torch_geometric.nn.conv.EGConv
   GraphGEN https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GENConv.html#torch_geometric.nn.conv.GENConv
   ```
   The edges in the GO database contain information about the relationship between genes, so we can use the below relation aware network for encoding
   ```bash
   RGCN https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv
   FastRGCN https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.FastRGCNConv.html#torch_geometric.nn.conv.FastRGCNConv
   RGATConvNet  https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGATConv.html#torch_geometric.nn.conv.RGATConv
   ```

   To run our code using the above mentioned network, we can just do the following
   ```bash
   bash submit.sh GCN RGCN
   ```
   You can replace GCN with (GAT,GATv2,EG,GraphGEN) and replace RGCN with (FastRGCN, RGATConvNet)

3. Visualization
   After running the model, we can visualize the loss curve and the AUROC, AUPRC curve using the following code. Assume that you have already run for all combinations of HumanNet encoder and GO database encoder
   ```bash
    python visualization/plot.py --input_dir runs  --output_dir visualization \
        --plot_auprc True --plot_loss True --plot_auroc True
   ```
   if you want to indiviually plot the result curve for one specific combination, you can run the follwowing code
   ```bash
    python visualization/plot_individual.py --g_encoder GCN \
       --kg_encoder RGCN --input_dir runs  --output_dir visualization \
        --plot_auprc True --plot_loss True --plot_auroc True
   ```
We alos provide a jupyter notebook `notebook.ipynb` for you to explore the pipeline of this project.
We provide `format_code.sh` for formatting the code in this repo.
## Acknowledgement
This codebase is built upon the project https://github.com/yhchen1123/CoGO