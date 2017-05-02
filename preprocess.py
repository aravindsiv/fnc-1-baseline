import cPickle as pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
from collections import defaultdict
from preprocess_ import preprocess_func
from main_file import train_classifier,test_classifier
import csv
import numpy as np
import unicodedata
import cPickle as pickle
def _normalize(text):
    return unicodedata.normalize('NFKD', text.decode('utf8')).encode('ascii', 'ignore')
reverse_dict={1:"unrelated",0:"related"}

class PreProcessor:
    def __init__(self,splits_folder="splits/",data_folder="fnc-1/"):
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
                
        self.train_data = []
        self.test_data = []
        self.complete_train=[]
        self.complete_test=[]
        for i in bodies:
            for j in range(len(headlines[i])):
                    if i in train_ids:
                        self.complete_train.append([_normalize(bodies[i]),_normalize(headlines[i][j]),_normalize(stances[i][j]),i])
                        if stances[i][j] != "unrelated":
                             self.train_data.append([_normalize(bodies[i]),_normalize(headlines[i][j]),_normalize(stances[i][j]),i])
                    else:
                        self.complete_test.append([_normalize(bodies[i]),_normalize(headlines[i][j]),_normalize(stances[i][j]),i])
                        if stances[i][j] != "unrelated":
                             self.test_data.append([_normalize(bodies[i]),_normalize(headlines[i][j]),_normalize(stances[i][j]),i])
                        
        self.train_data = np.array(self.train_data)
        self.test_data = np.array(self.test_data)
        self.complete_train = np.array(self.complete_train)
        self.complete_test = np.array(self.complete_test)        
        
        print "Number of training examples: %s" %(len(self.train_data))
        print "Number of test examples: %s" %(len(self.test_data))
        print "Number of full training examples: %s" %(len(self.complete_train))
        print "Number of full test examples: %s" %(len(self.complete_test))        

    def preprocess_keras(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(np.vstack([self.train_data[:,0:2],self.test_data[:,0:2]]).flatten())

        stances = set(self.train_data[:,2])
        
        self.label_index = {} # labels_index["agree"]
        
        for i,j in enumerate(stances):
            self.label_index[j] = i

        print "Found %s unique tokens" %(len(self.tokenizer.word_index))
    
    def first_stage_model(self,file_name):
        classifier=pickle.load(open(file_name,"rb"))
        return classifier
        
    def first_stage_predicition(self,test_data, file_name,model_type=0):
        if(model_type==0): #0 for logidtic svm
            classifier=pickle.load(open(file_name,"rb"))
        else:
            classifier=load_model(file_name)
        test_labels,test_data=preprocess_func(test_data)
        pred_labels,normalized_test_labels=test_classifier(classifier,test_data,test_labels)
        filtered_test_data=[]
        for i,j in enumerate(pred_labels):
            if(model_type!=0):
                buff=np.argmax(j)
                j=reverse_dict[buff]
            if(j!="unrelated"):      
                 filtered_test_data.append(test_data[i])
        filtered_test_data=np.array(test_data)    
        return filtered_test_data
        
        
    def preprocess_stageone(self):
        train_labels,train_data=preprocess_func(self.complete_train)
        test_labels,test_data=preprocess_func(self.complete_test)
        #classifier=train_classifier(train_data,train_labels)
        #pred_labels,normalized_test_labels=test_classifier(classifier,test_data,test_labels)
        

        pickle.dump(train_labels,open("training_label.pk","wb"))
        pickle.dump(train_data,open("train_data.pk","wb"))
        pickle.dump(test_labels,open("test_label.pk","wb"))
        pickle.dump(test_data,open("test_data.pk","wb"))

        #print "Found %s unique tokens" %(len(self.tokenizer.word_index)) 
    
    def preprocess_stageone_noreturn(self,train_data,test_data):
        train_labels,train_data=preprocess_func(train_data)
        test_labels,test_data=preprocess_func(test_data)
        #classifier=train_classifier(train_data,train_labels)
        #pred_labels,normalized_test_labels=test_classifier(classifier,test_data,test_labels)
        return(train_labels,train_data,test_labels,test_data)
       

    def make_data_fold(self,k,splits_folder="splits/"):
        '''This function uses training_ids_k.txt as the cross-validation data, and the other files for training that
        particular model.'''
        train_data_k = []
        test_data_k = []

        test_ids = []
        with open(splits_folder+"training_ids_"+str(k)+".txt") as f:
            for l in f:
                test_ids.append(int(l))

        for i in range(self.train_data.shape[0]):
            if int(self.train_data[i,3]) in test_ids:
                test_data_k.append(self.train_data[i,0:3])
            else:
                train_data_k.append(self.train_data[i,0:3])

        train_data_k = np.array(train_data_k)
        test_data_k = np.array(test_data_k)

        print "Number of training examples for fold %s: %s" %(k,len(train_data_k))
        print "Number of test examples for fold %s: %s" %(k,len(test_data_k))

        return train_data_k, test_data_k
    
    
    def make_data_fold_stageone(self,k,splits_folder="splits/"):
        '''This function uses training_ids_k.txt as the cross-validation data, and the other files for training that
        particular model, for stage one'''
        train_data_k = []
        test_data_k = []

        test_ids = []
        with open(splits_folder+"training_ids_"+str(k)+".txt") as f:
            for l in f:
                test_ids.append(int(l))

        for i in range(self.complete_train.shape[0]):
            if int(self.complete_train[i,3]) in test_ids:
                test_data_k.append(self.complete_train[i,0:4])
            else:
                train_data_k.append(self.complete_train[i,0:4])

        train_data_k = np.array(train_data_k)
        test_data_k = np.array(test_data_k)

        print "Number of training examples for fold %s: %s" %(k,len(train_data_k))
        print "Number of test examples for fold %s: %s" %(k,len(test_data_k))
        return train_data_k, test_data_k

        return train_data_k, test_data_k
    def make_data_keras(self,fold,mode="train"):
        train_data_k, test_data_k = self.make_data_fold(fold)
                
        bodies_sequence = self.tokenizer.texts_to_sequences(train_data_k[:,0])
        headlines_sequence = self.tokenizer.texts_to_sequences(train_data_k[:,1])

        bodies_sequence_test = self.tokenizer.texts_to_sequences(test_data_k[:,0])
        headlines_sequence_test = self.tokenizer.texts_to_sequences(test_data_k[:,1])
        
        max_seq_length = 400 #max(max([len(bodies_sequence[i]) for i in range(len(bodies_sequence))]),\
                                #max([len(bodies_sequence_test[i]) for i in range(len(bodies_sequence_test))]))

        bodies_data = pad_sequences(bodies_sequence,maxlen=max_seq_length)
        headlines_data = pad_sequences(headlines_sequence,maxlen=max_seq_length)

        bodies_data_test = pad_sequences(bodies_sequence_test,maxlen=max_seq_length)
        headlines_data_test = pad_sequences(headlines_sequence_test,maxlen=max_seq_length)
        
        labels = np.zeros((len(train_data_k),1))
        labels_test = np.zeros((len(test_data_k),1))
        
        for i in range(len(train_data_k)):
            labels[i] = self.label_index[train_data_k[i,2]]

        for i in range(len(test_data_k)):
            labels_test[i] = self.label_index[test_data_k[i,2]]

        labels = to_categorical(labels)
        labels_test = to_categorical(labels_test)
        
        print "Shape ofs bodies data tensor: " +str(bodies_data.shape)
        print "Shape of headlines data tensor: " +str(headlines_data.shape)
        print "Shape of labels tensor: " + str(labels.shape)

        if mode == "train":
            return {"train":[bodies_data, headlines_data, labels],"test":[bodies_data_test, headlines_data_test, labels_test],"max_seq_length":max_seq_length}
        else:
            return {"bodies":bodies_data_test,"headlines":headlines_data_test,"labels":labels_test}


    def get_embedding_matrix(self,we_file="fnc-1/glove.6B.300d.txt"):
        embeddings_index = {}
        
        f = open(we_file)
        
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        
        print "Found %s word vectors." % len(embeddings_index)
            
        self.embedding_matrix = np.zeros((len(self.tokenizer.word_index)+1, 300)) # Change this

        
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

        return self.embedding_matrix

if __name__ == "__main__":
    pp = PreProcessor()
    #pp.preprocess_keras()
    pp.preprocess_stageone()

    
    #_ = pp.make_data_keras(0)
    
