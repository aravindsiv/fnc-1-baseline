import json
import numpy as np
from preprocess import _normalize

file_name_train = "snli_1.0_train.json"
file_name_test = "snli_1.0_test.json"
file_name_dev = "snli_1.0_dev.json"


def read_json_data():
    """
    Read json data (per preprocess logic) for train, test and dev json.
    :param file_name: 
    :return: train_data, test_data, dev_data
    """
    with open(file_name_train) as data_file:
        data_rows = json.load(data_file)
        train_data = [[_normalize(row["sentence1"]), _normalize(row["sentence2"]), _normalize(row["gold_label"])] for row in data_rows]

    with open(file_name_test) as data_file:
        data_rows = json.load(data_file)
        test_data = [[_normalize(row["sentence1"]), _normalize(row["sentence2"]), _normalize(row["gold_label"])] for row in data_rows]

    with open(file_name_dev) as data_file:
        data_rows = json.load(data_file)
        dev_data = [[_normalize(row["sentence1"]), _normalize(row["sentence2"]), _normalize(row["gold_label"])] for row in data_rows]

    return np.array(train_data), np.array(test_data), np.array(dev_data)

# train_d, test_d, dev_d = read_json_data()
# print(train_d[0])
# print(test_d[0])
# print(dev_d[0])
