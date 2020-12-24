"""ALL the config needed for this experiment."""


class Config:
    # numeric constants
    MIN_SAMPLES_PER_INTENT = 50
    TRAIN_SPLIT = 0.7
    MIN_NGRAM_COUNT = 16

    # filepaths
    RAW_DATA_FILEPATH = "./data/training_data_for_query_classification.pkl"
    TRAINING_DATA_FILEPATH = "./data/training_data.json"
    TESTING_DATA_FILEPATH = "./data/testing_data.json"

    #
    LOG_FORMAT = "%(asctime)s %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s"
    DATE_FORMAT = "%Y-%m-%d:%H:%M:%S"


config = Config()
