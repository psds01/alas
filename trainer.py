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


class OptimizationStrategy:
    name = "OptimizationStrategy"

    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train_dataset: Dataset,
        test_dataset: Dataset,
        n_epochs: int,
        top_frac: float = 1.0,
    ):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_epochs = n_epochs
        self.top_frac = top_frac

    def add_loss_to_dataset(self):
        pass

    def get_optimized_dataset(self):
        pass

    def optimize(self):
        pass

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
