import logging
import os
import typing
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


class NNModel(nn.Module):
    def __init__(self, n_features: int, n_hidden: int, n_classes: int):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


def get_net_criterion_optimizer(
    n_features: int, n_hidden: int, n_classes: int, init_filepath: os.PathLike
):
    """
    Utility to define and INITIALIZE the model
    with the same weights for different runs.
    """
    logger.info("Defining model.")
    net = NNModel(n_features, n_hidden, n_classes)
    # if path does not exist, save and load (redundant!)
    if not os.path.exists(init_filepath):
        logger.info("Saved weights do not exist, saving model weights.")
        torch.save(net.state_dict(), init_filepath)
    logger.info("Initializing model with saved weights.")
    net.load_state_dict(torch.load(init_filepath))

    logger.info("Creating Criterion.")
    criterion = nn.CrossEntropyLoss()
    logger.info("Creating Optimizer.")
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer.zero_grad()
    logger.info("Done.")
    return net, criterion, optimizer
