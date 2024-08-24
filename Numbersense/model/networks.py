"""Contains all possible models that can be used in this project."""

from os import getenv
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray


class PrintLayer(nn.Module):
    def __init__(self, verbose: bool = True):
        super(PrintLayer, self).__init__()
        self.verbose = verbose

    def forward(self, x: ndarray):
        if self.verbose:
            print(x.shape)
        return x


def randomize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


class SiameseActionClassificationNet(nn.Module):
    """SiameseActionClassificationNet - takes two images as input.

    :param Any embedding_net: The embedding network of the model.
    :param str unique_id: Spefifies whether the model uses pretrained weights or not.
    """

    def __init__(self, embedding_net: Any, unique_id: str, xavier_init: bool = True):
        super(SiameseActionClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.embedding_dimension = self.embedding_net.embedding_dimension
        self.unique_id = unique_id
        self.name: str = self.unique_id
        self.nonlinear = nn.PReLU()  # Activation function
        self.fc1 = nn.Linear(self.embedding_dimension * 2, 256)  # 2 embedding-inputs
        # self.fc2 = nn.Linear(256, 384)
        # self.fc3 = nn.Linear(384, 512)
        # self.fc4 = nn.Linear(512, 680)
        # self.fc5 = nn.Linear(680, 384)
        # self.fc6 = nn.Linear(384, 3)  # 3 for either of the possible actions

        self.fclast = nn.Linear(256, 3)
        if xavier_init:
            torch.manual_seed(getenv('SEED', 0)) if getenv('SEED', 0) else None
            randomize_weights(self.fc1)
            randomize_weights(self.fclast)

    def forward(self, x1: ndarray, x2: ndarray):
        # Feature extraction stage
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        cat_out = torch.cat([output1, output2], dim=1)
        out = self.nonlinear(cat_out)
        # Classifier
        out = self.fc1(out)
        # out = self.nonlinear(out)
        # out = self.fc2(out)
        # out = self.nonlinear(out)
        # out = self.fc3(out)
        # out = self.nonlinear(out)
        # out = self.fc4(out)
        # out = self.nonlinear(out)
        # out = self.fc5(out)
        # out = self.nonlinear(out)
        # out = self.fc6(out)
        out = self.fclast(out)
        scores = F.log_softmax(out, dim=-1)
        return scores

    def get_embedding(self, x: ndarray):
        return self.embedding_net(x)


class ConcatenationNet(nn.Module):
    """ConcatenationNet - takes side-by-side concatenated images as input.

    :param Any embedding_net: The embedding network of the model.
    :param str unique_id: Spefifies whether the model uses pretrained weights or not.
    """

    def __init__(self, embedding_net: Any, unique_id: str):
        super(ConcatenationNet, self).__init__()
        self.embedding_net = embedding_net  # AlexNet + seq. layers
        self.embedding_dimension = self.embedding_net.embedding_dimension
        self.unique_id = unique_id
        self.name: str = (
            "concatenation_net-" + self.embedding_net.name + "-" + self.unique_id
        )
        self.nonlinear = nn.PReLU()  # Activation function
        self.fc1 = nn.Linear(self.embedding_dimension, 256)  # Only 1 embedding input
        self.fc2 = nn.Linear(256, 3)  # 3 for either of the actions

    def forward(self, x: ndarray):
        # Feature extraction stage
        out = self.embedding_net(x)
        out = self.nonlinear(out)
        out = self.fc1(out)
        out = self.fc2(out)
        scores = F.log_softmax(out, dim=-1)
        return scores

    def get_embedding(self, x: ndarray):
        return self.embedding_net(x)


class EmbeddingNet(nn.Module):
    """EmbeddingNet - takes one image as input."""

    def __init__(
        self,
        embedding_dimension: int,
        backbone: nn.Module,
        freeze_backbone: bool = True,
        concatenated: bool = False,
        xavier_init: bool = True,
    ):
        super(EmbeddingNet, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.backbone = backbone.to("cpu")

        # Freeze weights so that gradient won't be calculated for them!
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Conv2d := (in_channels, out_channels, kernel_size, stride=1, padding=0)
        # MaxPool2d := (kernel_size, stride=None, padding=0)
        self.convnet = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=(5, 5), padding=(2, 2)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        # Determine dims of the connecting layers
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 244, 244) if getenv("GRAYSCALE", 0) else torch.randn(1, 3, 244, 244)
            output = self.backbone.forward(dummy_input)
            _, c, _, _ = output.shape
            self.convnet[0] = nn.Conv2d(c, 256, kernel_size=(5, 5), padding=(2, 2))
            output = self.convnet(output)
            output = output.view(output.size()[0], -1)
            self.linear_input_dim = output.view(output.size(0), -1).shape[1]

        if concatenated:
            self.linear_input_dim /= 2

        # Input depends on the second level dim of output after view!
        # Linear := (in_features, out_features)
        self.fc = nn.Sequential(
            nn.Linear(self.linear_input_dim, self.embedding_dimension),
            nn.PReLU(),
            nn.Linear(self.embedding_dimension, self.embedding_dimension),
        )

        if xavier_init:
            torch.manual_seed(getenv('SEED', 0)) if getenv('SEED', 0) else None
            randomize_weights(self.fc)
            randomize_weights(self.convnet)

    def forward(self, x: ndarray):
        output = self.backbone.forward(x)
        output = self.convnet(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x: ndarray):
        return self.forward(x)


# Source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class ClassifierNet(nn.Module):
    """ClassifierNet - takes one image as input."""

    def __init__(self):
        super(ClassifierNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(53824, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x: ndarray):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
