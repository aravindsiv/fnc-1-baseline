from preprocess import PreProcessor
from keras.models import load_model
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
import sys
from score import report_score
sys.path.append("./utils")
np.set_printoptions(threshold=np.nan)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='no | lstm | lstm_1 | gru', default='logistic')
    parser.add_argument('-model_path')
    args = vars(parser.parse_args())
    base_string="./models/stage_one4"
    model_type=args["model"]
    path = args['model_path']
    flag=0
    base_string+=model_type
    if(model_type=="nn"):
        flag=1
        base_string+=".h5"
    pp = PreProcessor()
    pp.preprocess_keras()
    test_data = pp.complete_test
    actual_labels=test_data[:,2]
    filtered_test_data, filtered_test_labels, global_labels = pp.first_stage_predicition(test_data,base_string,flag) # 0 for logistic, svm , 1 for Neural Network


    model = load_model(path)
    bodies = filtered_test_data[:,0]
    headlines = filtered_test_data[:,1]
    stances = filtered_test_labels

    bodies, headlines, stances = pp.make_data_test(bodies,headlines,stances)

    y = np.argmax(model.predict([bodies,headlines]),axis=1)

    pred_ctr = 0

    for i in range(global_labels.shape[0]):
        if global_labels[i] == "related":
            global_labels[i] = pp.rev_index[y[pred_ctr]]
            pred_ctr += 1

    report_score(actual_labels, global_labels)

