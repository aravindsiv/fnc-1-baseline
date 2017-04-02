import rules_lib_ml as rules_lib
from sklearn.metrics import confusion_matrix
import pickle
import types
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


func_list=[rules_lib.__dict__.get(a).__name__ for a in dir(rules_lib) if isinstance(rules_lib.__dict__.get(a), types.FunctionType)]
def feature_extractor(data_point,max_length):
    obs = []
    for m in func_list:
        func = getattr(rules_lib, m)
        obs.append(func(data_point,max_length))
    return obs
def max_len(train_data,key):
    max_length=0
    for m in train_data:
        buff=len(m[key])
        if(buff>max_length):
            max_length=buff
    return max_length

def max_len2(train_data,key):
    max_length=[]
    for m in train_data:
        max_length.append(len(m[key]))

    return sum(max_length)/len(max_length)
def train_classifier(dataset,labels,training_ids):
    classifier = LogisticRegression(penalty='l1')
    training_data=[]
    training_labels=[]
    train_data=[]
    training_index=[]
    max_hd=0
    for i,k in enumerate(dataset):
        print(i)
        if(k["id"] not in training_ids):
            continue
        train_data.append(k)
        training_index.append(labels[i])
    max_length=max_len(train_data, "hd_tok")
    for i,k in enumerate(train_data):
        obs=feature_extractor(k,max_length)
        training_data.append(obs)
        #current_label=labels[i]
        current_label = training_index[i]
        if(current_label in ["agree","discuss","disagree"]):
            current_label="related"
        training_labels.append(current_label)
    print("Training Started")
    print(training_data[:3])
    training_vector=np.array(training_data)
    training_labels=np.array(training_labels)
    classifier.fit(training_vector,training_labels)
    return classifier,max_length

def test_classifier(dataset,classifier,labels,test_ids,max_length):

    test_data=[]
    test_labels=[]
    test_body=[]
    test_hd=[]

    for i,k in enumerate(dataset):
        if(k["id"] not in test_ids):
            continue
        test_body.append(" ".join([text for text in k["body_tok"] if(len(text.strip())>0)]))
        test_hd.append(k["hd"])
        obs=feature_extractor(k,max_length)
        test_data.append(obs)
        current_label=labels[i]
        if(current_label in ["agree","discuss","disagree"]):
            current_label="related"
        test_labels.append(current_label)
    test_vector=np.array(test_data)
    pred_labels=classifier.predict(test_vector)
    return test_labels,pred_labels,test_data,test_hd,test_body

dataset=pickle.load(open("dataset_pickle.pk","rb"))
labels=pickle.load(open("labels.pk","rb"))
training_ids=[int(k.strip()) for k in open("/home/ubuntu/ml_project/fnc-1-baseline/splits/training_ids.txt","rb")]
test_ids=[int(k.strip()) for k in open("/home/ubuntu/ml_project/fnc-1-baseline/splits/hold_out_ids.txt","rb")]


classifier,max_length=train_classifier(dataset,labels,training_ids)
pickle.dump(classifier,open("trained_classifier.pk","wb"))
print("Done Training")
actual_labels,pred_labels,test_data,test_hd,test_body=test_classifier(dataset,classifier,labels,test_ids,max_length)

a="HD|Body|Pred|Actual|"
for m in func_list:
    a+=m+"|"
file_output=open("logger_file.txt","w")
logger=file_output.write
logger("%s\n"%a)
for i,k in enumerate(pred_labels):
    if(actual_labels[i]!=k):
        logger("%s|%s|%s|%s|%s|%s|%s\n"%(test_hd[i],test_body[i],k,actual_labels[i],test_data[i][0],test_data[i][1],test_data[i][2]))
file_output.close()

print(accuracy_score(actual_labels, pred_labels))
print("related number",actual_labels.count("related"),"\n")
print("unrelated number",actual_labels.count("unrelated"),"\n")
print(confusion_matrix(actual_labels, pred_labels))

