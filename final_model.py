from preprocess import PreProcessor
from keras.models import load_model
import numpy as np
import argparse
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='no | lstm | lstm_1 | gru', default='logistic')
    args = vars(parser.parse_args())
    base_string="./models/stage_one4"
    model_type=args["model"]
    flag=0
    base_string+=model_type
    if(model_type=="nn"):
        flag=1
        base_string+=".h5"
    pp = PreProcessor()
    test_data = pp.complete_test
    filtered_test_data = pp.first_stage_predicition(test_data,base_string,flag) # 0 for logistic, svm , 1 for Neural Network
    print "got filtered data"
    print filtered_test_data.shape
    model = load_model('lstm_2.h5')
    bodies = filtered_test_data[:,0]
    headlines = filtered_test_data[:,1]
    stances = filtered_test_data[:,2]
    y = model.predict([bodies,headlines])
