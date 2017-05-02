from preprocess import PreProcessor
from keras_models import test_model
from sklearn.metrics import pairwise
from keras.models import Sequential, load_model

if __name__ == "__main__":
    pp = PreProcessor()
    model_name = "lstm.h5"
    data_dict = {}
    pp.preprocess_keras()
    data_dict = pp.make_data_keras(4)
    bodies, headlines, labels = data_dict["train"]
    bodies_t, headlines_t, labels_t = data_dict["test"]

    model = load_model("models_snli/"+model_name)
    loss, acc = model.evaluate([bodies_t, headlines_t], labels_t)
    print "Test loss: %s, accuracy: %s" % (loss, acc)