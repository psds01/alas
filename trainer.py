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


class OptimizationStrategy(object):
    pass


class BaseStrategy(OptimizationStrategy):
    pass


class TopPopulationStrategy(OptimizationStrategy):
    pass


class TopPercentageStrategy(OptimizationStrategy):
    pass
