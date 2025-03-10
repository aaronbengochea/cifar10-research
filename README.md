# Model inspiration 
|-------------------------------------------------------| Test Acc% | Hidden Acc% |
| [ResNet18](https://arxiv.org/abs/1512.03385)          |  95.32%   |   84.97%    |

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## General
- Data: stores the CIFAR10 training and test data
- Testset: stores the hidden kaggle CIFAR10 testset used in the classification competition
- Experiments: an assortment of different experiments and their details ran on variants of ResNets with param count below 5M
- kaggle.py: notebook to be imported and used in kaggle enviornment for recreation of model initialization, training, and inference
- resnet.py: contains the implementation of the ResNet, and BasicBlock classes
- training.py: 









