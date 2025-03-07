# Model inspiration 
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |


## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## General
- Evals: stores each model evaluation vs the hidden testset
- Checkpoint: stores each the best overall models checkpoint
- Data: stores the CIFAR10 training and test data
- History: stores the training history of each model
- Models: stores each model architecture implementation and its varients 
- Testset: stores the hidden kaggle CIFAR10 testset
- main.py: the main training file
- parameters.ipynb: a quick script to count the total and trainable parameters of each model
- predictions.py: a script to make predictions on the hidden kaggle testset

## TODO
- [ ] Change checkpoint setup so that we can track checkpoints for distinct architectures as opposed to overall best model
- [ ] Setup argparse for predictions.py so that we can pass which model we'd like to use for predictions as a command line argument
- [ ] General argparse setup for LR, Epochs, etc. (gather inspo from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py)
- [ ] We will need to explore smaller architectures since vanilla resnet18 with [2,2,2,2] layers yields ~11M params, challange limit is 5M



