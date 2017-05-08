# -*- coding: utf-8 -*-
"""
Created on Sun april  2 18:09:09 2017

@author: ubuntu

This file is used to generate distribution graphs for the features
"""


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import argparse
import sys
#from main_file import train_classifier, test_classifier
import numpy as np
import cPickle as pickle
import rules_lib_ml as rules_lib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import types
from preprocess_ import preprocess_func
import csv
import numpy as np

import unicodedata
import cPickle as pickle
import matplotlib

func_list=[rules_lib.__dict__.get(a).__name__ for a in dir(rules_lib) if isinstance(rules_lib.__dict__.get(a), types.FunctionType)]
def feature_extractor(data_point,max_length):
    obs = []
    for m in func_list:
        func = getattr(rules_lib, m)
        obs.append(func(data_point,max_length))
    return obs
def _normalize(text):
    return unicodedata.normalize('NFKD', text.decode('utf8')).encode('ascii', 'ignore')

def initial(splits_folder="splits/",data_folder="fnc-1/"):
        bodies = {}
        headlines = defaultdict(list)
        stances = defaultdict(list)
        
        train_ids = []
        holdout_ids = []

        with open(data_folder+"train_bodies.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                bodies[int(row[0])] = row[1]
                
        with open(data_folder+"train_stances.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                headlines[int(row[1])].append(row[0])
                stances[int(row[1])].append(row[2])
                
        with open(splits_folder+"training_ids.txt") as f:
            for l in f:
                train_ids.append(int(l))
        
        with open(splits_folder+"hold_out_ids.txt") as f:
            for l in f:
                holdout_ids.append(int(l))
                

        complete_train=[]
        complete_test=[]
        for i in bodies:
            for j in range(len(headlines[i])):
                    if i in train_ids:
                        complete_train.append([_normalize(bodies[i]),_normalize(headlines[i][j]),_normalize(stances[i][j]),i])
                        
                    else:
                        complete_test.append([_normalize(bodies[i]),_normalize(headlines[i][j]),_normalize(stances[i][j]),i])
                        
        #self.train_data = np.array(self.train_data)
        #self.test_data = np.array(self.test_data)
        complete_train = np.array(complete_train)
        complete_test = np.array(complete_test)        
        return(complete_train,complete_test)
        #print "Number of training examples: %s" %(len(self.train_data))
        #print "Number of test examples: %s" %(len(self.test_data))
        #print "Number of full training examples: %s" %(len(self.complete_train))
        #print "Number of full test examples: %s" %(len(self.complete_test))        
def preprocess_stageone(complete_train,complete_test):
        train_labels,train_data=preprocess_func(complete_train)
        test_labels,test_data=preprocess_func(complete_test)
        #classifier=train_classifier(train_data,train_labels)
        #pred_labels,normalized_test_labels=test_classifier(classifier,test_data,test_labels)
        
        print("done")  
        pickle.dump(train_labels,open("training_label.pk","wb"))
        pickle.dump(train_data,open("train_data.pk","wb"))
        pickle.dump(test_labels,open("test_label.pk","wb"))
        pickle.dump(test_data,open("test_data.pk","wb"))
if __name__ == "__main__":
   
    #train_data,test_data=initial()
    #preprocess_stageone(train_data,test_data)

            
        
    train_labels=pickle.load(open("training_label.pk","rb"))
    train_data=pickle.load(open("train_data.pk","rb"))
    training_labels=[]
    for i,k in enumerate(train_labels):
        current_label=train_labels[i]                
        if(current_label in ["agree","discuss","disagree"]):
                current_label="related"
        training_labels.append(current_label)
    pickle.dump(training_labels,open("training_labels.pk","wb"))
    print func_list        
    #max_length=0
    #feature_dict=[]
    #for i,k in enumerate(train_data):
    #    print i
    #    obs=feature_extractor(k,max_length)
    #    feature_dict.append(obs)
        
    #feature_dict=np.array(feature_dict)
    #pickle.dump(feature_dict,open("feature_dict","wb"))
    feature_dict=pickle.load(open("feature_dict","rb"))
    #print(set(train_labels))
    unnrelated_features=[]
    related_features=[]
    for i , k in enumerate(training_labels):
        if(training_labels[i]=="unrelated"):
            unnrelated_features.append(feature_dict[i])
            continue
        related_features.append(feature_dict[i])
    unnrelated_features=np.array(unnrelated_features)
    
    related_features=np.array(related_features)
    feat_name={0:'Adjective', 1:'Noun', 2:'Verb'}
    list_iter=[0,0,1,1,2,2]
    for i,k in enumerate(list_iter):
        print i
        #first_feature=list(unnrelated_features[:,k])
        feat_string=feat_name[k]
        #second_feature=list(related_features[:,2]) 
        if(i%2==0):
            buff=related_features
            class_string="related"
            col="g"
        else:
            buff=unnrelated_features
            class_string="unrelated"
            col="r" 
        first_feature=list(buff[:,k])    
        plt.hist(first_feature, bins=50, histtype='stepfilled', color=col, label=class_string.title())
        plt.title("Histogram for Common_"+feat_string+"_Feature for "+class_string.title())
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(feat_string+"_"+class_string)
        plt.close()
        #fig.savefig('path/to/save/image/to.png')
            
    """
    df = pd.DataFrame(dict(x=first_feature, y=second_feature, label=train_labels))
    groups = df.groupby('label')
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
    ax.legend()
    plt.show()
    """
    #colors = ['red','green']

    
    #fig = plt.figure(figsize=(8,8))
    #plt.scatter(first_feature, second_feature, c=train_labels, cmap=matplotlib.colors.ListedColormap(colors))
    #cb = plt.colorbar()
    #loc = np.arange(0,max(label),max(label)/float(len(colors)))
    #cb.set_ticks(loc)
    #cb.set_ticklabels(colors)        
    #train_labels,train_data,test_labels,test_data=pp.preprocess_stageone_noreturn(train_data,test_data)
    #path_file="models/"
    #file_name="stage_one"+str(fold)+model_type
    #classifier=train_classifier(train_data,train_labels,path_file+file_name,model_type)
    #pred_labels,test_labels=test_classifier(classifier,test_data,test_labels,max_length=0)
    #if(model_type in ["nn"]):