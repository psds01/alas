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
    def __init__(self, config: Config):
        """
        Inputs:
        -------
            config: Config object
        """
        self.config = config
        self.feature_map = {}
        self.label_map = {}

    def preprocess(self, text: Text) -> Text:
        """
        1. lower case 
        2. Replace all digits by 0
        3. Split at special characters
        """
        text = str(text).lower()
        text = re.sub(r"\d", "0", text)
        return " ".join(re.split(r"\W+", text)).strip()

    def get_ngrams(self, text: Text) -> Dict:
        grams = defaultdict(int)
        text = self.preprocess(text)
        # collect unigrams
        for ch in text:
            grams[ch] += 1

        # collect bigrams
        for chs in zip(text, text[1:]):
            grams["".join(chs)] += 1

        return grams

    def featurize(self, text: Text) -> np.ndarray:
        arr = np.zeros(len(self.feature_map))
        grams = self.get_ngrams(text)
        for grm, count in grams.items():
            if grm in self.feature_map:
                arr[self.feature_map[grm]] += count
        return arr

    def encode_label(self, label: Text) -> np.ndarray:
        return np.array([self.label_map[label]])

    def load_from_file(self):
        logger.info("Loading feature map.")
        self.feature_map = json.load(open(self.config.FEATURE_MAP_FILEPATH, "r"))
        logger.info("Loading label map.")
        self.label_map = json.load(open(self.config.LABEL_MAP_FILEPATH, "r"))
        logger.info("Featurizer attributes loaded.")
        return True

    def save_to_file(self):
        logger.info("Saving feature map.")
        json.dump(self.feature_map, open(self.config.FEATURE_MAP_FILEPATH, "w"))
        logger.info("Saving label map.")
        json.dump(self.label_map, open(self.config.LABEL_MAP_FILEPATH, "w"))
        logger.info("Saved featurizer attributes.")
        return True

    def train(self, datasets: List[Dataset]):
        logger.info("Training featurizer.")
        tracker = defaultdict(int)
        labels = set()
        for dataset in datasets:
            for instance in tqdm(dataset):
                labels.add(instance.intent)
                n_grams = self.get_ngrams(instance.text)
                for gram, count in n_grams.items():
                    tracker[gram] += count

        logger.info("N-grams and labels reviewed.")

        tracker = {
            gram: count
            for gram, count in tracker.items()
            if count >= self.config.MIN_NGRAM_COUNT
        }
        tracker = sorted(tracker.key())
        tracker = {gram: index for index, gram in enumerate(tracker)}
        self.feature_map = tracker
        logger.info("Feature map created.")

        labels = sorted(labels)
        labels = {label: index for index, label in enumerate(labels)}
        self.label_map = labels
        logger.info("Label map created.")

        self.save_to_file()
