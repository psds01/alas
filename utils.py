"""Data Preparation Utilities."""

import json
import logging
import os
import pickle
import typing
from collections import defaultdict
from typing import Any, Dict, List, Text, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

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
        for item in dataset:
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
