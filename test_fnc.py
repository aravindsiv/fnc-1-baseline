from preprocess import PreProcessor
from keras_models import test_model
from sklearn.metrics import pairwise
from keras.models import Sequential, load_model

if __name__ == "__main__":
    pp = PreProcessor()
    model_name = "lstm.h5"
    data_dict_0 = {}
    data_dict_1 = {}
    data_dict_2 = {}
    data_dict_3 = {}
    data_dict_4 = {}
    pp.preprocess_keras()
    data_dict_0 = pp.make_data_keras(0)
    data_dict_1 = pp.make_data_keras(1)
    data_dict_2 = pp.make_data_keras(2)
    data_dict_3 = pp.make_data_keras(3)
    data_dict_4 = pp.make_data_keras(4)

    data_dict = data_dict_0.update(data_dict_1)
    data_dict = data_dict.update(data_dict_2)
    data_dict = data_dict.update(data_dict_3)
    data_dict = data_dict.update(data_dict_4)

    bodies, headlines, labels = data_dict["train"]
    bodies_t, headlines_t, labels_t = data_dict["test"]

    model = load_model("models_snli/"+model_name)
    loss, acc = model.evaluate([bodies_t, headlines_t], labels_t)
    print "Test loss: %s, accuracy: %s" % (loss, acc)