"""ALL the config needed for this experiment."""


class Config:
    # numeric constants
    MIN_SAMPLES_PER_INTENT = 50
    TRAIN_SPLIT = 0.7
    MIN_NGRAM_COUNT = 16
    HIDDEN_DIM = 256
    SAVE_EVERY = 5

    # filepaths
    RAW_DATA_FILEPATH = "./data/training_data_for_query_classification.pkl"
    TRAINING_DATA_FILEPATH = "./data/training_data.json"
    TESTING_DATA_FILEPATH = "./data/testing_data.json"
    FEATURE_MAP_FILEPATH = "./data/feature_map.json"
    LABEL_MAP_FILEPATH = "./data/label_map.json"
    INIT_MODEL_PATH = "./init_model.pt"
    BASE_CKPTS_DIR = "./ckpts"

    #
    LOG_FORMAT = "%(asctime)s %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s"
    DATE_FORMAT = "%Y-%m-%d:%H:%M:%S"


config = Config()
