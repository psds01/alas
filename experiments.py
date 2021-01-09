import json
import logging
import os
import pathlib
from typing import Any, Dict, List, Text

from tqdm import tqdm

from config import Config
from featurizer import Featurizer
from model import get_net_criterion_optimizer
from trainer import BaseStrategy, TopPercentageStrategy, TopPopulationStrategy
from utils import create_train_test_data, get_train_test_datasets

logger = logging.getLogger(__name__)


class Experiment:
    def __init__(self, name: Text, config: Dict):
        self.name = name
        logger.info("Creating config for the experiment.")
        c = Config()
        for key, value in config.items():
            setattr(c, key, value)

        # filepaths and init paths
        c.TRAINING_DATA_FILEPATH = os.path.join(
            c.BASE_CKPTS_DIR, self.name, "data", "training_data.json"
        )
        c.TESTING_DATA_FILEPATH = os.path.join(
            c.BASE_CKPTS_DIR, self.name, "data", "testing_data.json"
        )

        c.FEATURE_MAP_FILEPATH = os.path.join(
            c.BASE_CKPTS_DIR, self.name, "data", "feature_map.json"
        )
        c.LABEL_MAP_FILEPATH = os.path.join(
            c.BASE_CKPTS_DIR, self.name, "data", "label_map.json"
        )
        c.INIT_MODEL_PATH = os.path.join(c.BASE_CKPTS_DIR, self.name, "init.ckpt")
        c.FINAL_MODEL_PATH = os.path.join(c.BASE_CKPTS_DIR, self.name, "final.ckpt")

        self.config = c
        logger.info("Created config for the experiment.")

    def save_config(self):
        path = os.path.join(self.config.BASE_CKPTS_DIR, self.name, "config.json")
        # if os.path.exists(path):
        #     return True
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(vars(self.config), open(path, "w"))
        return True

    def run(self):
        # create dataset as per min count and split ratio
        create_train_test_data(self.config)

        # get datasets and featurize them
        datasets = get_train_test_datasets(self.config)
        train_dataset, test_dataset = datasets
        featurizer = Featurizer(self.config)
        featurizer.load_or_train(datasets)
        featurizer.featurize_datasets(datasets)

        # model
        n_features = len(featurizer.feature_map)
        n_hidden = self.config.HIDDEN_DIM
        n_classes = len(featurizer.label_map)
        init_filepath = self.config.INIT_MODEL_PATH
        experiment_id = self.name

        # save_every = self.config.SAVE_EVERY
        # self.config.SAVE_EVERY = save_every
        # n_epochs = 3

        # self.config.n_epochs = n_epochs
        n_epochs = self.config.n_epochs
        self.config.n_features = n_features
        self.config.n_classes = n_classes

        self.save_config()

        for top_frac in tqdm(range(9, 0, -1)):
            top_frac = round(0.1 * top_frac, 1)
            logger.info("\n\nRunning expt with top frac = {}".format(top_frac))
            for trainer in [BaseStrategy, TopPopulationStrategy, TopPercentageStrategy]:

                net, criterion, optimizer = get_net_criterion_optimizer(
                    n_features=n_features,
                    n_hidden=n_hidden,
                    n_classes=n_classes,
                    init_filepath=init_filepath,
                )

                params = [
                    experiment_id,
                    net,
                    criterion,
                    optimizer,
                    train_dataset,
                    test_dataset,
                    n_epochs,
                    top_frac,
                    self.config,
                ]

                trainer = trainer(*params)
                trainer.train()
