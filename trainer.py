import json
import logging
import os
import pathlib
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
        config: Config = config,
    ):
        """
        `experiment_id` :The initial source of randomness 
                    remains the same throughout an experiment.
        `top_frac` :for the same run of experiment (aka same source or randomness
                    aka model is initialized with the same weights), different %
                    tile/age population
        """
        self.experiment_id = experiment_id
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_epochs = n_epochs
        self.top_frac = top_frac
        self.config = config

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
        # NOTE: if your dataset does not fit into the memory, create batches
        # and then optimize the loss
        dataset = self.get_optimized_dataset()
        loss = sum(x.loss for x in dataset) / len(dataset)
        logger.info("Loss during training of {} = {}".format(self.name, loss))
        loss.backward()
        self.optimizer.step()
        logger.info("Network optimized.")

    def save_weights(self, basename: Text):
        params = self.net.state_dict()
        for key, value in params.items():
            filepath = os.path.join(
                self.config.BASE_CKPTS_DIR,
                self.experiment_id,
                self.top_frac,
                self.name,
                "weights",
                basename,
                "{}.pkl".format(key),
            )
            filepath = pathlib.Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(value, open(filepath, "wb"))

    def save_stats(
        self,
        basename: Text,
        train_y: np.ndarray,
        test_y: np.ndarray,
        train_pred: np.ndarray,
        test_pred: np.ndarray,
    ) -> None:
        stats = {
            "train_y": train_y.tolist(),
            "test_y": test_y.tolist(),
            "train_pred": train_pred.tolist(),
            "test_pred": test_pred.tolist(),
        }
        filepath = os.path.join(
            self.config.BASE_CKPTS_DIR,
            self.experiment_id,
            self.top_frac,
            self.name,
            "stats",
            "{}.pkl".format(basename),
        )
        filepath = pathlib.Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(stats, open(filepath, "wb"))
        logger.info("stats saved at: {}".format(str(filepath)))

    def evaluate(self, epoch: Union[int, Text], save: bool):
        train_X = torch.stack([x.feature for x in self.train_dataset])
        test_X = torch.stack([x.feature for x in self.test_dataset])

        # reshape to create a single batch
        # NOTE: feel free to change this if your data does not fit into memory
        train_X = train_X.view(train_X.shape[0], -1)
        test_X = test_X.view(test_X.shape[0], -1)

        # get labels
        train_y = torch.stack([x.label for x in self.train_dataset]).view(-1).numpy()
        test_y = torch.stack([x.label for x in self.test_dataset]).view(-1).numpy()

        # get prediction from the model
        train_pred = self.net(train_X)
        test_pred = self.net(test_X)

        # get the predicted class indices
        train_pred = torch.argmax(train_pred, axis=1).numpy()
        test_pred = torch.argmax(test_pred, axis=1).numpy()

        if save:
            self.save_stats(epoch, train_y, test_y, train_pred, test_pred)

    def train(self):
        logger.info("Training the network with {} strategy.".format(self.name))
        for epoch in tqdm(range(self.n_epochs)):
            epoch += 1

            # zero grad
            self.optimizer.zero_grad()

            # collect losses for all instances
            self.add_loss_to_dataset()

            # optimize the weights
            self.optimize()

            # is it time to save
            to_save = epoch % self.config.SAVE_EVERY == 0

            # save weights or not
            if to_save:
                self.save_weights(epoch)

            # eval on train and test dataset
            self.evaluate(epoch, to_save)


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
