# Model inspiration 
This repository takes inspiration from [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
We explore model architectures inspires by ResNet18

# Architecture Details 
| Model Name | Blocks per Layer | Channels per Layer | Parameter Count |
|------------|------------------|--------------------|-----------------|
| ResNet40   | [5, 7, 4, 3]     | [32, 64, 128, 256] | 4.99M           |

# Training and Performance Details
| Model Name | Test Set Acc% | Hidden Test Set Acc% | Optimizer                     | Scheduler                        |
|------------|---------------|----------------------|-------------------------------|----------------------------------|
| ResNet40   | 95.32%        | 84.97%               | SGD(lr=0.1, momentum=0.9)       | CosineAnnealingLR(Tmax=250)        |



### General Setup - Kaggle




## General
- Data: stores the CIFAR10 training and test data
- Testset: stores the hidden kaggle CIFAR10 testset used in the classification competition
- Experiments: an assortment of different experiments and their details ran on variants of ResNets with param count below 5M
- kaggle.py: notebook to be imported and used in kaggle enviornment for recreation of model initialization, training, and inference
- resnet.py: contains the implementation of the ResNet, and BasicBlock classes
- training.py: 




## Training
```
WORK IN PROGRESS

# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```






