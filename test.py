import logging

from config import config
from featurizer import Featurizer
from utils import get_train_test_datasets

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        format=config.LOG_FORMAT, datefmt=config.DATE_FORMAT, level=logging.INFO,
    )

    training_dataset, testing_dataset = get_train_test_datasets(config)
    for instance in training_dataset:
        pass
    logger.info(instance)
    for instance in testing_dataset:
        pass
    logger.info(instance)

    featurizer = Featurizer(config)
    featurizer.save_to_file()
    featurizer.load_from_file()
