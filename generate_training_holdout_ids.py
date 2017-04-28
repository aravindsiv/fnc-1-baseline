from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds


def main():
    d = DataSet()
    kfold_split(d, n_folds=5)

if __name__ == "__main__":
    main()