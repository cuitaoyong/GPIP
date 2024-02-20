# GPIP


This is the official implementation for the paper: "GPIP: Geometry-enhanced Pre-training on Interatomic Potentials".


- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [How to run this code](#how-to-run-this-code)

# Overview
Machine learning interatomic potentials (MLIPs) enables molecular dynamics (MD) simulations with ab initio accuracy and has been applied to various fields of physical science. However, the performance and transferability of MLIPs are limited by insufficient labeled training data, which require expensive ab initio calculations to obtain the labels, especially for complex molecular systems. To address this challenge, we design a novel geometric structure learning paradigm that consists of two stages. We first generate a large quantity of 3D configurations of target molecular system with classical molecular dynamics simulations. Then, we propose geometry-enhanced self-supervised learning consisting of masking, denoising, and contrastive learning to better capture the topology and 3D geometric information from the unlabeled 3D configurations. We evaluate our method on various benchmarks ranging from small molecule datasets to complex periodic molecular systems with more types of elements. The experimental results show that the proposed pre-training method can greatly enhance the accuracy of MLIPs with few extra computational costs and works well with different invariant or equivariant graph neural network architectures. Our method improves the generalization capability of MLIPs and helps to realize accurate MD simulations for complex molecular systems.


# System Requirements
## Hardware requirements

GPU required for running this code base, and NVIDIA A100 card and one RTX 3090 card has been tested.

## Software requirements
### OS Requirements
This code base is supported for *Linux* and has been tested on the following systems:
+ Linux: Ubuntu 20.04

### Python Version

Python 3.9.15 has been tested.

# Installation Guide:

### Install dependencies
```
conda install mamba -n base -c conda-forge
mamba env create -f environment.yaml
conda activate GPIP
```

- Install DGL from [here](https://www.dgl.ai/pages/start.html).
```
pip install  dgl -f https://data.dgl.ai/wheels/cu116/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

- `sudo`, if required

# How to run this code:

### Notebook (Demo)

In `GPIP.ipynb`, you can see the running time of pre-training stage and fine-tuning stage, as well as the replication of paper results, and you can also re-train the model through this Notebook.

### Download the pre-train dataset to Pretrain_dataset and save processed files to Pretrain_dataset:

```
python xyz2pt.py
```

### Pre-training on MD17

The model is pre-trained on the MD17 dataset, which consists of MD trajectories of small organic molecules with reference values of energy and forces calculated by ab initio molecular dynamics (AIMD) simulations. We use a pre-training dataset of 160,000 configurations sampled from CMD trajectories of eight different organic molecules.

```
python pretraining.py
```

### Fun-tuning on MD17 Datasets(Benzene2017 Dataset)

```
python finetune.py
```

For other datasets in MD17 Datasets, you can change the `data_name` in the 15 line of the `finetune.py` (replace the 'benzene2017' with 'aspirin', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic', 'toluene', 'uracil').


