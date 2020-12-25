import json
import logging
import os
import pickle
import re
import typing
from collections import Counter, defaultdict
from typing import Any, Dict, List, Text, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config, config
from utils import Dataset, Instance, get_train_test_datasets

logger = logging.getLogger(__name__)


def add_loss_to_dataset(dataset: Dataset, net: nn.Module, criterion: nn.Module) -> None:
    for instance in dataset:
        feature, label = instance.feature, instance.label
        output = net(feature)
        loss = criterion(output, label)
        instance.loss = loss


class OptimizationStrategy:
    name = "OptimizationStrategy"

    def __init__(
        self,
        experiment_id: Text,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train_dataset: Dataset,
        test_dataset: Dataset,
        n_epochs: int,
        top_frac: float = 1.0,
    ):
        self.experiment_id = experiment_id
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_epochs = n_epochs
        self.top_frac = top_frac

    def save_losses(self):
        return NotImplementedError

    def add_loss_to_dataset(self):
        logger.info("{}: Adding loss to training instances".format(self.name))
        add_loss_to_dataset(
            dataset=self.train_dataset, net=self.net, criterion=self.criterion
        )

        logger.info("{}: Adding loss to testing instances".format(self.name))
        add_loss_to_dataset(
            dataset=self.test_dataset, net=self.net, criterion=self.criterion
        )

    def get_optimized_dataset(self):
        return NotImplementedError

    def optimize(self):
        # get dataset/instances to optimize the model with.
        dataset = self.get_optimized_dataset()
        loss = sum(x.loss for x in dataset) / len(dataset)
        logger.info("Loss during training of {} = {}".format(self.name, loss))
        loss.backward()
        self.optimizer.step()
        logger.info("Network optimized.")

    def save_weights(self, name):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass


class BaseStrategy(OptimizationStrategy):
    name = "BaseStrategy"

    def __init__(self, *args):
        super().__init__(*args)


class TopPopulationStrategy(OptimizationStrategy):
    name = "TopPopulationStrategy"

    def __init__(self, *args):
        super().__init__(*args)


class TopPercentageStrategy(OptimizationStrategy):
    name = "TopPercentageStrategy"

    def __init__(self, *args):
        super().__init__(*args)
