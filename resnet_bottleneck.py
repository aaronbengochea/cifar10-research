import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, skip_kernel_size, stride, expansion_factor):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=skip_kernel_size, padding=skip_kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * expansion_factor, kernel_size=skip_kernel_size, padding=skip_kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels * expansion_factor)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * expansion_factor: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion_factor, kernel_size=skip_kernel_size, stride=stride, padding=skip_kernel_size//2, bias=False),
                nn.BatchNorm2d(out_channels * expansion_factor)
            )
            
        

    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x += self.shortcut(identity)
        x = F.relu(x)
        return x


class ResNetBottleneck(nn.Module):
    def __init__(self, blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, pool_size, starting_input_channels, expansion_factor, name, num_classes=10):
        super(ResNetBottleneck, self).__init__()
        self.in_channels = channels_per_layer[0]
        self.name = name
        self.expansion_factor = expansion_factor

        self.input_layer = nn.Sequential(
            nn.Conv2d(starting_input_channels, self.in_channels, kernel_size=kernels_per_layer[0], stride=1, padding=kernels_per_layer[0]//2, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )

        self.residual_layers = self._make_layers(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, expansion_factor)

        # Calculate the final output channel size
        final_channels = channels_per_layer[-1] * expansion_factor
        
        # Calculate the feature map size after pooling
        feature_size = pool_size * pool_size
        
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            nn.Linear(final_channels * feature_size, num_classes)
        )


    def _make_layer(self, out_channels, num_blocks, kernel_size, skip_kernel_size, stride, expansion_factor):
        blocks = []
        strides = []
        for i in range(num_blocks):
            current_stride = stride if i == 0 else 1
            if i == 0:
                strides.append(current_stride)
                
            blocks.append(BottleneckBlock(self.in_channels, out_channels, kernel_size, skip_kernel_size, current_stride, expansion_factor))
            self.in_channels = out_channels * expansion_factor
        
        if strides:
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
        x = F.relu(self.input_layer(x))
        x = self.residual_layers(x)
        x = self.output_layer(x)
        return x


def create_bottleneck_model(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, pool_size, starting_input_channels, expansion_factor, name, num_classes=10):
    return ResNetBottleneck(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, pool_size, starting_input_channels, expansion_factor, name, num_classes)


if __name__ == "__main__":
    model = create_bottleneck_model(
        blocks_per_layer = [2, 3, 2, 1],
        channels_per_layer = [50, 100, 200, 400],
        kernels_per_layer = [3, 3, 3, 3],
        skip_kernels_per_layer = [1, 1, 1, 1],
        pool_size = 1,
        starting_input_channels = 3,
        expansion_factor = 3,
        name = 'ResNet_v1'
    )
    
    print('Total model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    summary(model, (3, 32, 32))