# TODO develop UNET for classification
#   Remove the last layer for embedding (Same for VGG)
#   https://saturncloud.io/blog/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch/
import torch
from torch import nn

from src.utils.setup_logger import logger


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()

        # Encoder (contracting path)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder (expansive path)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        self.linear = nn.LazyLinear(num_classes)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return self.linear(x2)

class SimpleCNN(nn.Module):
    # 128 - 3 + 2 = 127 + 1 = 128
    # 128 - 2 / = 126 / 2 = 63+1=64
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16 * 16, num_classes)
        # self.fc1 = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        logger.debug(f'size equal to {x.shape}')
        # TODO: the point here is to transform it to [1,5], check how critron is calculated
        x = x.view(1, -1)
        logger.debug(f'size equal to {x.shape}')
        x = self.fc1(x)
        return x

