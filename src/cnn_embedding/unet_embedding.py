import warnings

from torch import nn
from torchvision import ops
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

from src.utils.setup_logger import logger

warnings.filterwarnings("ignore")


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()

        # Encoder (contracting path)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Decoder (expansive path)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        self.softmax = nn.Softmax()
        self.linear = nn.LazyLinear(num_classes)

    def forward(self, x, device="cuda"):
        # logger.debug(f"the input shape {x.shape}")
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        # logger.debug(f"shape before view {x2.shape}")
        x3 = x2.view(1, -1)
        lazy = nn.LazyLinear(32).to(device=device)
        x3 = lazy(x3)
        # logger.debug(f"shape after view {x3.shape}")
        x4 = self.linear(x3)
        return self.softmax(x4)


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
        logger.debug(f"size equal to {x.shape}")
        # TODO: the point here is to transform it to [1,5], check how critron is calculated

        logger.debug(f"size equal to {x.shape}")
        x = self.fc1(x)
        return x


class EfficientNetV2MultiClass(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetV2MultiClass, self).__init__()
        self.num_classes = num_classes
        weights = EfficientNet_V2_L_Weights.DEFAULT
        self.pretrained_eff_v2 = efficientnet_v2_l(weights=weights)

        self.pretrained_eff_v2.classifier[1].weight = nn.Parameter(self.pretrained_eff_v2.classifier[1].weight[:self.num_classes])
        self.pretrained_eff_v2.classifier[1].bias = nn.Parameter(self.pretrained_eff_v2.classifier[1].bias[:self.num_classes])

        self.pretrained_eff_v2.features[0] = nn.Sequential(
            ops.Conv2dNormActivation(
                1,
                out_channels=32,
                kernel_size=(2, 2),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
                # norm_layer = nn.BatchNorm2d,
                activation_layer=nn.SiLU,
            )
        )

        # Modify the classifier head for your specific number of classes
        # self.pretrained_unet.classifier[4] = nn.Conv2d(128, num_classes, kernel_size=(1, 1))
        self.softmax = nn.Softmax()

    def forward(self, x, device="cuda"):
        # Forward pass through the pretrained U-Net model
        self.pretrained_eff_v2.classifier[1] = nn.LazyLinear(
            self.num_classes, device=device
        )
        # logger.debug(f"x shape before {x.shape}")
        output = self.pretrained_eff_v2(x)
        # logger.debug(f"x shape after {output.shape}")

        return self.softmax(output)
