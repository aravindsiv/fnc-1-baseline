import rules_lib_ml as rules_lib
from sklearn.metrics import confusion_matrix
import pickle
import types
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


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

def train_classifier(train_data,train_labels,file_name,option="logistic"):
    print (func_list)
    if(option=="logistic"):
         classifier = LogisticRegression(penalty='l1')
    elif(option=="svm"):
        classifier = svm.SVC()
    training_data=[]
    training_labels=[]
    #train_data=[]
    #training_labels=[k for k in train_labels]
    max_hd=0
    #for i,k in enumerate(dataset):
    #    #print(i)
    #    if(k["id"] not in training_ids):
    #        continue
    #    train_data.append(k)
    #    training_index.append(labels[i])
    max_length=max_len(train_data, "hd_tok")
    for i,k in enumerate(train_data):
        obs=feature_extractor(k,max_length)
        #print (i)
        training_data.append(obs)
        #current_label=labels[i]
        current_label = train_labels[i]
        if(current_label in ["agree","discuss","disagree"]):
            current_label="related"
        training_labels.append(current_label)
    print("Training Started")
    print(training_data[:3])
    training_vector=np.array(training_data)
    training_labels=np.array(training_labels)
    if option == "nn":
        model = Sequential()
        model.add(Dense(4,input_dim=3))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.fit(training_vector,training_labels,epochs=5,batch_size=32)
        model.save(file_name) # File should be a .h5 file.
        return model
    else:
        classifier.fit(training_vector,training_labels)
        pickle.dump(classifier,open(filename,"wb"))
        return classifier

def test_classifier(classifier,test_data,labels,max_length=0):

    testing_data=[]
    test_labels=[]
    test_body=[]
    test_hd=[]

    for i,k in enumerate(test_data):
        #if(k["id"] not in test_ids):
        #    continue
        test_body.append(" ".join([text for text in k["body_tok"] if(len(text.strip())>0)]))
        test_hd.append(k["hd"])
        obs=feature_extractor(k,max_length)
        testing_data.append(obs)
        current_label=labels[i]
        if(current_label in ["agree","discuss","disagree"]):
            current_label="related"
        test_labels.append(current_label)
    test_vector=np.array(testing_data)
    pred_labels=classifier.predict(test_vector)
    # model.predict(test_vector) # if you are using keras
    #return test_labels,pred_labels,testing_data,test_hd,test_body
    return pred_labels,test_labels

#dataset=pickle.load(open("dataset_pickle1.pk","rb"))
#labels=pickle.load(open("labels1.pk","rb"))

#training_ids=[int(k.strip()) for k in open("/home/ubuntu/ml_project/fnc-1-baseline/splits/training_ids.txt","rb")]
#test_ids=[int(k.strip()) for k in open("/home/ubuntu/ml_project/fnc-1-baseline/splits/hold_out_ids.txt","rb")]

#print("Done Training")
#actual_labels,pred_labels,test_data,test_hd,test_body=test_classifier(test_data,classifier,test_label,max_length)

#a="HD|Body|Pred|Actual|"
#for m in func_list:
#    a+=m+"|"
#file_output=open("logger_file.txt","w")
#logger=file_output.write
#logger("%s\n"%a)
#for i,k in enumerate(pred_labels):
#    if(actual_labels[i]!=k):        
#        logger("%s|%s|%s|%s|%s|%s|%s\n"%(test_hd[i],test_body[i],k,actual_labels[i],test_data[i][0],test_data[i][1],test_data[i][2]))
#file_output.close()

#print(accuracy_score(actual_labels, pred_labels))
#print("related number",actual_labels.count("related"),"\n")
#print("unrelated number",actual_labels.count("unrelated"),"\n")
#print(confusion_matrix(actual_labels, pred_labels))

if __name__ == "__main__":
    training_label=pickle.load(open("training_label.pk","rb"))
    train_data=pickle.load(open("train_data.pk","rb"))
    test_label=pickle.load(open("test_label.pk","rb"))
    test_data=pickle.load(open("test_data.pk","rb"))
    #classifier=train_classifier(dataset,labels,training_ids)
    classifier=train_classifier(train_data,training_label)
    pickle.dump(classifier,open("trained_classifier.pk","wb"))