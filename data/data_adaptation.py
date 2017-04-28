import json

file_name = "snli_1.0_test.json"

with open(file_name) as data_file:
    data_rows = json.load(data_file)

    d = [[row["gold_label"], row["sentence1"], row["sentence2"]] for row in data_rows]

    # test the first row.
    print(d[0])
    print(len(d))