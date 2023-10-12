import torch
import torch.nn as nn
import torch.nn.functional as F


# define ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # defines two consecutive convolutional layers within the residual block
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut, in order to maintain consistency with the structure of the two convolutional layers' results, some processing is required here.
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):
        out = self.left(x)
        # Adding the outputs of the two convolutional layers to the processed x achieves the fundamental structure of ResNet
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # for CIFAR10, use 3x3 kernel with stride 1, incase of information loss
            # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False),   # original paper uses 7x7 kernel with stride 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # make layers
        self.layer1 = self.make_layer(ResBlock, channels=64, num_blocks=2, stride=1)
        self.layer2 = self.make_layer(ResBlock, channels=128, num_blocks=2, stride=2)
        self.layer3 = self.make_layer(ResBlock, channels=256, num_blocks=2, stride=2)
        self.layer4 = self.make_layer(ResBlock, channels=512, num_blocks=2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    # This function is primarily used to repeat the same residual block
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # The first residual block of each layer needs to be processed by the shortcut
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
