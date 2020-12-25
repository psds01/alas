import logging

from config import config
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

    # get dataset and featurizer
    datasets = get_train_test_datasets(config)
    train_dataset, test_dataset = datasets
    featurizer = Featurizer(config)
    featurizer.load_from_file()
    featurizer.featurize_datasets(datasets)

    # model
    n_features = len(featurizer.feature_map)
    n_hidden = config.HIDDEN_DIM
    n_classes = len(featurizer.label_map)
    init_filepath = config.INIT_MODEL_PATH
    net, criterion, optimizer = get_net_criterion_optimizer(
        n_features=n_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
        init_filepath=init_filepath,
    )

    experiment_id = "simple"
    n_epochs = 2
    top_frac = 0.5

    params = [
        experiment_id,
        net,
        criterion,
        optimizer,
        train_dataset,
        test_dataset,
        n_epochs,
        top_frac,
        config,
    ]

    trainer = BaseStrategy(*params)
    trainer.train()
    trainer = TopPopulationStrategy(*params)
    trainer.train()
    trainer = TopPercentageStrategy(*params)
    trainer.train()
