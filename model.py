import typing
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NNModel(nn.Module):
    def __init__(self, n_features: int, n_hidden: int, n_classes: int):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


def get_net(n_features, n_hidden, n_classes, init_filepath):
    """
    Utility to define and INITIALIZE the model
    with the same weights for different runs.
    """
    net = NNModel(n_features, n_hidden, n_classes)
    net.load_state_dict(torch.load(init_filepath))
    return net
