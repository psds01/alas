import datetime
import json
import logging

from tqdm import tqdm

from config import config
from experiments import Experiment
from featurizer import Featurizer
from model import get_net_criterion_optimizer
from trainer import BaseStrategy, TopPercentageStrategy, TopPopulationStrategy
from utils import get_train_test_datasets

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # TODO: init from different random weights for different experiments

    logging.basicConfig(
        format=config.LOG_FORMAT, datefmt=config.DATE_FORMAT, level=logging.INFO,
    )

    config = {
        "MIN_SAMPLES_PER_INTENT": 50,
        "TRAIN_SPLIT": 0.7,
        "MIN_NGRAM_COUNT": 16,
        "HIDDEN_DIM": 256,
        "SAVE_EVERY": 1,
        "RAW_DATA_FILEPATH": "./data/training_data_for_query_classification.pkl",
        "BASE_CKPTS_DIR": "./history",
        "LOG_FORMAT": "%(asctime)s %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
        "DATE_FORMAT": "%Y-%m-%d:%H:%M:%S",
        "n_epochs": 256,
    }

    params = [
        [50, 0.6, 16, 128],
        [50, 0.7, 16, 512],
        [50, 0.8, 16, 128],
        [50, 0.8, 16, 512],
        [50, 0.9, 16, 128],
    ]

    for i, param in enumerate(params):
        MIN_SAMPLES_PER_INTENT, TRAIN_SPLIT, MIN_NGRAM_COUNT, HIDDEN_DIM = param
        config["MIN_SAMPLES_PER_INTENT"] = MIN_SAMPLES_PER_INTENT
        config["MIN_NGRAM_COUNT"] = MIN_NGRAM_COUNT
        config["HIDDEN_DIM"] = HIDDEN_DIM
        config["TRAIN_SPLIT"] = TRAIN_SPLIT

        dt = datetime.datetime.now()
        experiment_name = "experiment_{}_{}".format(
            i + 1, dt.isoformat().replace("-", "_").replace(":", "_").split(".")[0]
        )
        expt = Experiment(experiment_name, config)
        expt.run()
