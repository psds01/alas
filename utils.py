"""Data Preparation Utilities."""

import json
import logging
import os
import pathlib
import pickle
import typing
from collections import defaultdict
from typing import Any, Dict, List, Text, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from config import Config, config

logger = logging.getLogger(__name__)


class Instance(object):
    """A learning instance from the dataset: a sample."""

    def __init__(self, text: Text, intent: Text):
        self.text = text
        self.intent = intent
        # loss from this instance
        self.loss = 0

    def __str__(self) -> Text:
        return "<< Text={},\nIntent={},\nloss={}>>".format(
            self.text, self.intent, self.loss
        )

    def __repr__(self) -> Text:
        return self.__str__()


class Dataset(object):
    """A dataset object: a list of instances with few functionalities."""

    def __init__(self, filepath: os.PathLike):
        """
        Inputs:
            filepath: filepath where the data is stored in `[(text, intent),..., ]` format 
        """
        self.dataset = []
        dataset = json.load(open(filepath, "r"))
        for item in tqdm(dataset):
            self.dataset.append(Instance(text=item[0], intent=item[1]))

    def __len__(self) -> int:
        return len(self.dataset)

    def __str__(self) -> Text:
        return "Dataset: len={}".format(len(self))

    def __repr__(self) -> Text:
        # Bad practice
        return self.__str__()

    def __iter__(self):
        for instance in self.dataset:
            yield instance


def get_train_test_datasets(config: Config) -> List[Dataset]:
    logger.info("Creating training dataset.")
    training_dataset = Dataset(config.TRAINING_DATA_FILEPATH)
    logger.info("Creating testing dataset.")
    testing_dataset = Dataset(config.TESTING_DATA_FILEPATH)
    logger.info(
        "len train dataset = {}, len of test dataset = {}".format(
            len(training_dataset), len(testing_dataset)
        )
    )
    return training_dataset, testing_dataset


def create_filepath(path: os.PathLike) -> os.PathLike:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def create_train_test_data(config: Config) -> None:
    """
    Inputs:
    -------
        config : the complete Config object
    """
    if os.path.exists(config.TRAINING_DATA_FILEPATH):
        logger.info(
            "Data already exists. Skipping creation of train and test datasets."
        )
        return True

    logger.info("Reading data from the raw data file.")
    total_data = pickle.load(open(config.RAW_DATA_FILEPATH, "rb"))

    logger.info("Creating dict object from the raw dataset.")
    dataset = defaultdict(list)
    for item in total_data:
        dataset[item[1]].append(item[0])

    logger.info("Filtering the dataset for intents with fewer samples.")
    dataset = {
        intent: queries
        for intent, queries in dataset.items()
        if len(queries) >= config.MIN_SAMPLES_PER_INTENT
    }
    logger.info(
        "Creating train and test datasets with train split = {}".format(
            config.TRAIN_SPLIT
        )
    )
    train_data = []
    test_data = []
    for intent, queries in dataset.items():
        thres = int(config.TRAIN_SPLIT * len(queries))
        for i, query in enumerate(queries):
            if i < thres:
                train_data.append((query, intent))
            else:
                test_data.append((query, intent))

    # random shuffle and DO THIS ONLY ONCE  and keep this dataset constant throughout
    logger.info("Shuffling train and test datasets.")
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    train_path = create_filepath(config.TRAINING_DATA_FILEPATH)
    test_path = create_filepath(config.TESTING_DATA_FILEPATH)

    # train_path = pathlib.Path(train_path)
    # test_path = pathlib.Path(test_path)

    # train_path.parent.mkdir(parents=True, exist_ok=True)
    # test_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving train and test datasets at: {}".format(train_path.parent))
    json.dump(train_data, open(train_path, "w"))
    json.dump(test_data, open(test_path, "w"))
    logger.info("Done.")
    return True
