import json
import numpy as np
from preprocess import _normalize

file_name_train = "snli_1.0_train.json"
file_name_test = "snli_1.0_test.json"
file_name_dev = "snli_1.0_dev.json"


def read_json_data(src_folder="data/"):
    """
    Read json data (per preprocess logic) for train, test and dev json.
    :param file_name: 
    :return: train_data, test_data, dev_data
    """
    with open(src_folder+file_name_train) as data_file:
        data_rows = json.load(data_file)
        train_data = [[_normalize(row["sentence1"]), _normalize(row["sentence2"]), _normalize(row["gold_label"]), idx] for idx, row in enumerate(data_rows)]

    with open(src_folder+file_name_test) as data_file:
        data_rows = json.load(data_file)
        test_data = [[_normalize(row["sentence1"]), _normalize(row["sentence2"]), _normalize(row["gold_label"]), idx] for idx, row in enumerate(data_rows)]

    with open(src_folder+file_name_dev) as data_file:
        data_rows = json.load(data_file)
        dev_data = [[_normalize(row["sentence1"]), _normalize(row["sentence2"]), _normalize(row["gold_label"]), idx] for idx, row in enumerate(data_rows)]

    return np.array(train_data), np.array(test_data), np.array(dev_data)