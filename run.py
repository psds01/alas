import logging

from tqdm import tqdm

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
    experiment_id = "simple"
    save_every = 2
    config.SAVE_EVERY = save_every
    n_epochs = 128
    for top_frac in tqdm(range(9, 0, -2)):
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
                config,
            ]

            trainer = trainer(*params)
            trainer.train()