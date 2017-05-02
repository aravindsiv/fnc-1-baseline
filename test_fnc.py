from preprocess import PreProcessor
from keras_models import test_model
from sklearn.metrics import pairwise

if __name__ == "__main__":
    pp = PreProcessor()
    model_name = "lstm.h5"
    data_dict = {}
    data_dict["bodies"] = pp.test_data[:,0]
    data_dict["headline"] = pp.test_data[:, 1]
    data_dict["labels"] = pp.test_data[:, 2]
    test_model("models_snli/"+model_name, pp.test_data)