# -*- coding: utf-8 -*-
"""
Created on Mon May  1 09:31:51 2017

@author: ubuntu
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import argparse
import sys
from preprocess import PreProcessor
from main_file import train_classifier, test_classifier
import numpy as np
label_dict={"related":0,"unrelated":1}
reverse_dict={1:"unrelated",0:"related"}

def an():
    return an
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-fold', help='fold to train the model on', default='0')
    parser.add_argument('-model', help='no | lstm | lstm_1 | gru', default='no')
    args = vars(parser.parse_args())
    model_type=args['model']
    fold=int(args['fold'])
    
    pp = PreProcessor()   
    train_data, test_data=pp.make_data_fold_stageone(fold)
    train_labels,train_data,test_labels,test_data=pp.preprocess_stageone_noreturn(train_data,test_data)
    path_file="models/"
    file_name="stage_one"+str(fold)+model_type
    classifier=train_classifier(train_data,train_labels,path_file+file_name,model_type)
    pred_labels,test_labels=test_classifier(classifier,test_data,test_labels,max_length=0)
    new_dict=[]
    for k in pred_labels:
        buff=np.argmax(k)
        new_dict.append(reverse_dict[buff])
        
    #print(file_name,accuracy_score(test_labels, pred_labels))
    print new_dict
    print(file_name,accuracy_score(test_labels, new_dict))
    
