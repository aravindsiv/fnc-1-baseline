from preprocess import PreProcessor
from keras.models import load_model
import numpy as np

if __name__ == '__main__':
    pp = PreProcessor()
    test_data =pp.test_data
    filtered_test_data = pp.first_stage_predicition(test_data,'trained_classifier.pk',0) # 0 for logistic, svm , 1 for Neural Network
    model = load_model('my_model.h5')
    bodies = np.asarray(model[1])
    headlines = np.asarray(model[2])
    y = model.predict([bodies,headlines])