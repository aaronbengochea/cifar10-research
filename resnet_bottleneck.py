import torch
import torch.nn as nn
from torchsummary import summary

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, skip_kernel_size, stride):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=skip_kernel_size, stride=stride, padding=skip_kernel_size//2, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += self.shortcut(identity)
        x = nn.ReLU(inplace=True)(x)
        return x
    

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, skip_kernel_size, stride, expansion_factor):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=skip_kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * expansion_factor, kernel_size=skip_kernel_size, bias=False),
            nn.BatchNorm2d(out_channels * expansion_factor)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * expansion_factor: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion_factor, kernel_size=skip_kernel_size, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * expansion_factor)
            )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = nn.ReLU(x)
        x = self.conv2(x)
        x = nn.ReLU(x)
        x = self.conv3(x)
        x += self.shortcut(identity)
        x = nn.ReLU(inplace=True)(x)
        return x




class ResNetBottleneck(nn.Module):
    def __init__(self, blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, pool_size, starting_input_channels, expansion_factor, name, num_classes=10):
        super(ResNetBottleneck, self).__init__()
        self.in_channels = channels_per_layer[0]
        self.name = name


        self.input_layer = nn.Sequential(
            nn.Conv2d(starting_input_channels, self.in_channels, kernel_size=kernels_per_layer[0], stride=1, padding=kernels_per_layer[0]//2, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
            
        )

        self.residual_layers = self._make_layers(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, expansion_factor)


        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            nn.Linear(channels_per_layer[-1] * expansion_factor, num_classes)
        )


    def _make_layer(self, out_channels, num_blocks, kernel_size, skip_kernel_size, stride, expansion_factor):
        blocks = []
        strides = []
        for i in range(num_blocks):
            if i != 0:
                stride = 1

            blocks.append(BottleneckBlock(self.in_channels, out_channels, kernel_size, skip_kernel_size, stride, expansion_factor))
            strides.append(stride)
            self.in_channels = out_channels * expansion_factor
        
        print(strides, '\n')

        return nn.Sequential(*blocks)



    def _make_layers(self, blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, expansion_factor):
        layers = []
        for i in range(len(blocks_per_layer)):
            layers.append(
                self._make_layer(
                    out_channels = channels_per_layer[i],
                    num_blocks = blocks_per_layer[i],
                    kernel_size = kernels_per_layer[i],
                    skip_kernel_size = skip_kernels_per_layer[i],
                    stride = 1 if i == 0 else 2,
                    expansion_factor = expansion_factor
                )
            )

        return nn.Sequential(*layers)



    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_layers(x)
        x = self.output_layer(x)
        return x



def create_bottleneck_model(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, pool_size, starting_input_channels, expansion_factor, name):
    return ResNetBottleneck(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, pool_size, starting_input_channels, expansion_factor, name)


if __name__ == "__main__":
    model = create_bottleneck_model(
        blocks_per_layer = [1, 1, 3, 1],
        #channels_per_layer = [64, 128, 256, 512],
        channels_per_layer = [49, 98, 196, 392],
        kernels_per_layer = [3, 3, 3, 3],
        skip_kernels_per_layer = [1, 1, 1, 1],
        pool_size = 1,
        starting_input_channels = 3,
        expansion_factor = 3,
        name = 'ResNet_v1'
    )
    
    print('Total model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    summary(model, (3, 32, 32))

