from preprocess import PreProcessor
from keras_models import test_model

if __name__ == "__main__":
    pp = PreProcessor()
    model_name = "lstm.h5"
    test_model("models_snli/"+model_name, pp.test_data)