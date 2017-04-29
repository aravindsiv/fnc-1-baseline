import sys
sys.path.append("/home/ubuntu/ml_project/fnc-1-baseline/utils")

import spacy
import pickle
from generate_test_splits import *
import json
nlp = spacy.load('en')
from nltk.tag import StanfordNERTagger
st = StanfordNERTagger('/home/ubuntu/Downloads/stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz','/home/ubuntu/Downloads/stanford-ner-2016-10-31/stanford-ner.jar')


print ("he;knvlk klenvken")
from dataset import DataSet

dataset = DataSet()

folds,hold_out_ids=kfold_split(dataset)

def preprocess_func(dataset):
    articles=dataset.articles
    output={}
    data_proc=[]
    mapping_id={}
    vocab_tokens={}
    for i,k in enumerate(dataset.stances):
        if(i==100):
            break
        print(i)
        buff={}
        buff["id"]=k['Body ID']

        output[i]=k["Stance"]
        buff["body"]=articles[buff["id"]]
        buff["hd"]=k["Headline"]
        if(buff["id"] in mapping_id):
            buff["body_pos"]=data_proc[mapping_id[buff["id"]]]["body_pos"]
            buff["body_tok"]=data_proc[mapping_id[buff["id"]]]["body_tok"]
            buff["body_entities"]=data_proc[mapping_id[buff["id"]]]["body_entities"]
        else:
            mapping_id[buff["id"]]=i
            buff["body_pos"] =[]
            buff["body_tok"]=[]
            doc_body = nlp(buff["body"])
            entity_list=[]
            buff_list=[]
            sent_list=[]
            for span in doc_body.sents:
                sent = ''.join(doc_body[i].string for i in range(span.start, span.end)).strip()
                sent_list.append(sent.split())
            buff_list=st.tag_sents(sent_list)
            for m in buff_list:
                entity_list.extend(m)
            for word in doc_body:
                buff["body_pos"].append(word.pos_)
                buff["body_tok"].append(word.text)
            entity_dict={entity:set() for entity in ["PERSON", "ORGANIZATION", "LOCATION"]}
            
            for pair in entity_list:
                if(pair[1] in ["PERSON", "ORGANIZATION", "LOCATION"]):
                     entity_dict[pair[1]].add(pair[0])
            buff["body_entities"]=entity_dict     
        doc_hd=nlp(buff["hd"])

        buff["hd_pos"]=[]
        buff["hd_tok"]=[]
        for word in doc_hd:
            buff["hd_pos"].append(word.pos_)
            buff["hd_tok"].append(word.text)
        entity_dict={entity:set() for entity in ["PERSON", "ORGANIZATION", "LOCATION"]}
        #entity_list=st.tag(buff["hd_tok"])
        entity_list=[]
        buff_list=[]
        sent_list=[]
        for span in doc_hd.sents:
                sent = ''.join(doc_hd[i].string for i in range(span.start, span.end)).strip()
                #entity_list.extend(st.tag(sent.split()))
                sent_list.append(sent.split())
        buff_list=st.tag_sents(sent_list)
        for m in buff_list:
            entity_list.extend(m)        
        for pair in entity_list:
            if(pair[1] in ["PERSON", "ORGANIZATION", "LOCATION"]):
                entity_dict[pair[1]].add(pair[0])
        buff["hd_entities"]=entity_dict     
        vocab_tokens[i]=[tok.lower().strip() for tok in set(buff["hd_tok"]).union(set(buff["body_tok"]))]    
        data_proc.append(buff)    
    return output,data_proc,vocab_tokens

output,data_proc,vocab_tokens=preprocess_func(dataset)
pickle.dump(output,open("labels100.pk","wb"))
pickle.dump(data_proc,open("dataset_pickle100.pk","wb"))
with open('data.json100', 'w') as fp:
    json.dump(vocab_tokens, fp)






