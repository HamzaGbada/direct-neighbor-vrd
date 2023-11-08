import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, is_tensor, from_numpy
from torchvision import datasets, models
import numpy as np


class VGG(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        EBMBED_SIZE = 50
        nb_class = 26
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 8, 3, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 16, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 32, 2, stride=2, padding=1),
        #     nn.ReLU(True)
        # )

        self.flatten = nn.Flatten(start_dim=1)
        self.flatten2 = nn.Flatten(end_dim=1)
        # self.linear = nn.Linear(sh1*sh2*sh3, 100)
        self.lazy_linear2 = nn.Sequential(
            nn.LazyLinear(embed_size * 2, bias=False),
            # nn.ReLU(True),
            # nn.Linear(128, embed_size)
        )
        self.linear2 = nn.LazyLinear(nb_class)
        # self.model_ft = models.resnet18(pretrained=True)
        # num_ftrs = self.model_ft.fc.in_features
        # self.model_ft.fc = nn.Linear(num_ftrs, embed_size)
        # VGG start
        self.conv1 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=1),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, padding=1),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, padding=1),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, padding=1),
            nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, padding=1),
            nn.ReLU(True),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, images, word_embed, device=None):
        if not is_tensor(images):
            images = from_numpy(np.array(images))

        ### print("images.shape")
        ### print(images.shape)
        ### print("word_embed.shape")
        ### print(word_embed.shape)
        ### print("x.type")
        ### print(type(images))
        ### print("word_embed.type")
        ### print(type(word_embed))
        # images1 = self.cnn(images)
        EBMBED_SIZE = 50
        channels = 3
        #         print("images.shape")
        #         print(images.shape)
        #         print("len(images.shape)")
        #         print(len(images.shape))
        if len(images.shape) == 2:
            channels = 1  # single (grayscale)

        else:
            channels = images.shape[1]

        images = nn.Conv2d(
            in_channels=channels,
            out_channels=64,
            kernel_size=2,
            padding=1,
            device=device,
        )(images)
        images1 = self.conv1(images)
        images1 = self.conv2(images1)
        #         images1 = self.conv3(images1)
        #         images1 = self.conv4(images1)
        #         images1 = self.conv5(images1)
        images1 = self.max_pool(images1)
        ## Simple
        # images1 = self.cnn(images)
        ## Simple
        # images = self.model_ft(images)
        # images = self.model_fc(images)
        ### images = self.lazy_linear(images)
        ### print("cnn.shape")
        ### print(images1.shape)
        images = self.flatten(images1)
        ### print("flatten.shape")
        ### print(images.shape)
        images = nn.Linear(
            images1.shape[1] * images1.shape[2] * images1.shape[3],
            EBMBED_SIZE,
            device=device,
        )(images)
        ### print("lazy_linear.shape")
        ### print(images.shape)
        word_embed = torch.reshape(word_embed, (1, EBMBED_SIZE))
        images = torch.stack((word_embed, images), dim=1)
        ### print("stack shape BOBOBOBOBOBO: ", images.shape)
        images = self.lazy_linear2(images)

        images = self.flatten2(images)
        # print("flatten shape jlkfjmdjgkldfjglmksdfjglkdj: ", images.shape)
        images = self.linear2(images)
        # TODO the probelmm is from the stack perform reshape after stack ValueError: Expected input batch_size (2) to match target batch_size (1).
        # print("images41414.shape1")
        # print(images.shape)
        # print("word_embed.shape1")
        # print(word_embed.shape)
        # print("x.type1")
        # print(type(images))
        # print("word_embed.type1")
        # print(type(word_embed))
        # x = self.fc3(x)
        # print("images.shape1")
        # print(images.shape)
        images = F.sigmoid(images)
        # print("sigmoid.shape1")
        # print(images.shape)
        return images


net = Net(50)
