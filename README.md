# PyTorch implementation of Π-Nets

Chrysos, Grigorios G., et al. "Π-nets: Deep Polynomial Neural Networks." arXiv preprint arXiv:2003.03828 (2020).  
https://arxiv.org/abs/2003.03828

## Requirement
- pytorch
- pytorch-ignite
- matplotlib
- torchviz, graphviz (optional)

## Usage

```
usage: train.py [-h] [-o O] [--cg] [-b B] [-e E] [--weight_decay WEIGHT_DECAY] [--lr LR] [--betas BETAS BETAS]
optional arguments:
  -h, --help            show this help message and exit
  -o O                  output directory
  --cg                  visualize computational graph (requires torchviz, graphviz)
  -b B                  batch size
  -e E                  epoch
  --weight_decay WEIGHT_DECAY
                        weight decay
  --lr LR               learning rate
  --betas BETAS BETAS   beta1 and beta2 of Adam
```