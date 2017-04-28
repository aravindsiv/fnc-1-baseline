from dataset import DataSet
from generate_test_splits import kfold_split, get_stances_for_folds


d = DataSet()
kfold_split(d, n_folds=5)