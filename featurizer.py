import json
import logging
import os
import pickle
import re
import typing
from collections import Counter, defaultdict
from typing import Any, Dict, List, Text, Union

import numpy as np
from tqdm import tqdm

from config import Config, config
from utils import Dataset, Instance, get_train_test_datasets

logger = logging.getLogger(__name__)


class Featurizer(object):
    def __init__(self):
        pass

    def load_from_file(self):
        pass

    def preprocess(self):
        pass

    def get_ngrams(self):
        pass

    def train(self):
        pass

    def save_to_file(self):
        pass

    def featurize(self):
        pass

    def encode_label(self):
        pass
