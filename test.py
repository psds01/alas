import logging

from config import config
from featurizer import Featurizer
from model import get_net
from trainer import BaseStrategy, TopPercentageStrategy, TopPopulationStrategy
from utils import get_train_test_datasets

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        format=config.LOG_FORMAT, datefmt=config.DATE_FORMAT, level=logging.INFO,
    )

    datasets = get_train_test_datasets(config)
    training_dataset, testing_dataset = datasets
    for instance in training_dataset:
        pass
    logger.info(instance)
    for instance in testing_dataset:
        pass
    logger.info(instance)

    featurizer = Featurizer(config)
    # featurizer.train(datasets)
    # featurizer.save_to_file()
    featurizer.load_from_file()
    ft = featurizer.featurize("hello world")
    # logger.info(ft)
    featurizer.featurize_datasets(datasets)
    for dataset in datasets:
        for instance in dataset:
            pass

    logger.info(
        "\nText={}, \nintent={},\nfeature shape={},\nlabel={}.".format(
            instance.text, instance.intent, instance.feature.shape, instance.label
        )
    )

    logger.info((instance.feature.shape, instance.label.shape))

    # sizes
    n_features = len(featurizer.feature_map)
    n_hidden = config.HIDDEN_DIM
    n_classes = len(featurizer.label_map)
    init_filepath = config.INIT_MODEL_PATH
    net = get_net(
        n_features=n_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
        init_filepath=init_filepath,
    )

    logger.info("Testing Opt strats.")

    params = list(range(9))
    trainer = BaseStrategy(*params)

    def get_sample_print(trainer):
        return "::".join(
            map(
                str,
                [
                    trainer.name,
                    trainer.experiment_id,
                    trainer.net,
                    trainer.criterion,
                    trainer.optimizer,
                    trainer.train_dataset,
                    trainer.test_dataset,
                    trainer.n_epochs,
                    trainer.top_frac,
                    trainer.config,
                ],
            )
        )

    logger.info(get_sample_print(trainer))

    params = [x + 1 for x in params]
    trainer = TopPopulationStrategy(*params)
    logger.info(get_sample_print(trainer))

    params = [x + 5 for x in params]
    trainer = TopPercentageStrategy(*params)
    logger.info(get_sample_print(trainer))
