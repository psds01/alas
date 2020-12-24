"""Data Preparation Utilities."""

import json
import logging
import os
import pickle
import typing
from collections import defaultdict
from typing import Any, Dict, List, Text, Union

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

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __iter__(self):
        pass
