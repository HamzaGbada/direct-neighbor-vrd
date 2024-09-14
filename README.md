# Direct-Neighbor-VRD

This repository contains the official implementation of the paper titled [Information Extraction from Visually Rich Documents Using Directed Weighted Graph Neural Network](https://doi.org/10.1007/978-3-031-70552-6_15).

## Abstract

This paper introduces a novel approach for information extraction (IE) from visually rich documents (VRD) by employing a directed weighted graph representation. This approach enhances performance by capturing relationships among VRD components using directed weighted graphs, as opposed to traditional methods based on Euclidean distance. The IE task is treated as a node classification problem, with graph convolutional networks (GCNs) processing the VRD graphs. Evaluations conducted on five real-world datasets demonstrate the efficacy and alignment with established norms.

## Dependencies

To run the code, you need the following libraries:

- [DGL](https://www.dgl.ai/) (Deep Graph Library)
- [PyTorch](https://pytorch.org/) (Deep Learning Framework)
- [Python](https://www.python.org/) (Programming Language)
- [NetworkX](https://networkx.org/documentation/stable/tutorial.html) (Graph Library)
- [OpenCV-Python](https://opencv.org/) (Computer Vision Library)

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Building the Graph-based Dataset

To build a graph-based dataset, use the following command:

```bash
python builder.py build -d <dataset>
```

This command creates a graph-based dataset for node classification for a specific dataset.

**Optional Arguments:**

- `-d DATASET, --dataset DATASET`: Choose the dataset to use. Options are `XFUND`, `FUNSD`, `SROIE`, `Wildreceipt`, or `CORD`.
- `-n MAX_NODE, --max_node MAX_NODE`: Maximum number of nodes per node (edges per node). Default is 6.

**Example:**

```bash
python builder.py build -d CORD
```

### Training the Model

To train the model, use the following command:

```bash
python train.py -h
```

**Arguments:**

- `-d DATANAME, --dataname DATANAME`: Select the dataset for model training. Options are `FUNSD`, `SROIE`, `Wildreceipt`, or `CORD`.
- `-p PATH, --path PATH`: Path to the dataset for model training.
- `-hs HIDDEN_SIZE, --hidden_size HIDDEN_SIZE`: GCN hidden size. Default is 32.
- `-hl HIDDEN_LAYERS, --hidden_layers HIDDEN_LAYERS`: Number of GCN hidden layers. Default is 20.
- `-lr LEARNING_RATE, --learning_rate LEARNING_RATE`: Learning rate. Default is 0.01.
- `-e EPOCHS, --epochs EPOCHS`: Number of epochs. Default is 200.

**Example:**

```bash
python train.py -d CORD -hs 64 -hl 128
```

## Acknowledgments

We acknowledge the contributions of the authors of the paper and the developers of the libraries used in this project.