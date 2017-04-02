# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 18:10:06 2017

@author: ubuntu
"""

import sys
sys.path.append("/home/ubuntu/ml_project/fnc-1-baseline/utils")

import spacy
import pickle
from generate_test_splits import *
nlp = spacy.load('en')



from dataset import DataSet

dataset = DataSet()

folds,hold_out_ids=kfold_split(dataset)

def preprocess_func(dataset):
    articles=dataset.articles
    output={}
    data_proc=[]
    used_body=set()
    mapping_id={}
    for i,k in enumerate(dataset.stances):
        print(i)
        buff={}
        buff["id"]=k['Body ID']

        output[i]=k["Stance"]
        buff["body"]=articles[buff["id"]]
        buff["hd"]=k["Headline"]
        if(buff["id"] in mapping_id):
            buff["body_pos"]=data_proc[mapping_id[buff["id"]]]["body_pos"]
            buff["body_tok"]=data_proc[mapping_id[buff["id"]]]["body_tok"]
        else:
            mapping_id[buff["id"]]=i
            buff["body_pos"] =[]
            buff["body_tok"]=[]
            doc_body = nlp(buff["body"])
            for word in doc_body:
                buff["body_pos"].append(word.pos_)
                buff["body_tok"].append(word.text)

        doc_hd=nlp(buff["hd"])

        buff["hd_pos"]=[]
        buff["hd_tok"]=[]
        for word in doc_hd:
            buff["hd_pos"].append(word.pos_)
            buff["hd_tok"].append(word.text)
        data_proc.append(buff)
    print(len(output))
    return output,data_proc

output,data_proc=preprocess_func(dataset)
pickle.dump(output,open("labels1.pk","wb"))
pickle.dump(data_proc,open("dataset_pickle1.pk","wb"))
generate_hold_out_split (dataset)
jk=1





