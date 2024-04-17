import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=num_channels)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_features=num_channels)

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        return F.relu(out + x) # residual connection


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(num_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.block = Block(num_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))

        out = self.block(out)

        out = self.avgpool(out)

        out = out.view(out.size(0), -1) # flatten the output

        return self.fc(out)