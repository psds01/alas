import logging

from config import config
from featurizer import Featurizer
from model import get_net
from trainer import BaseStrategy, TopPercentageStrategy, TopPopulationStrategy
from utils import get_train_test_datasets

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # TODO: init from different random weights for different experiments
    pass
