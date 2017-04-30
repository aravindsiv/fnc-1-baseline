import cPickle as pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from collections import defaultdict
from preprocess_ import preprocess_func
from data.data_adaptation import read_json_data

import csv
import numpy as np
import unicodedata


def _normalize(text):
    return unicodedata.normalize('NFKD', text.decode('utf8')).encode('ascii', 'ignore')


class PreProcessor:

    def __init__(self, ):

        self.train_data, self.test_data, self.dev_data = read_json_data()

        print "Number of training examples: %s" %(len(self.train_data))
        print "Number of test examples: %s" % (len(self.test_data))
        print "Number of dev examples: %s" % (len(self.dev_data))

    def preprocess_keras(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(np.vstack([self.train_data[:,0:2],self.test_data[:,0:2]]).flatten())

        stances = set(self.train_data[:,2])
        
        self.label_index = {} # labels_index["agree"]
        
        for i,j in enumerate(stances):
            self.label_index[j] = i

        print "Found %s unique tokens" %(len(self.tokenizer.word_index))
        
    def preprocess_stageone(self):
        train_labels,train_data=preprocess_func(self.train_data)
        test_labels,test_data=preprocess_func(self.test_data)
        pickle.dump(train_labels,open("training_label.pk","wb"))
        pickle.dump(train_data,open("train_data.pk","wb"))
        pickle.dump(test_labels,open("test_label.pk","wb"))
        pickle.dump(test_data,open("test_data.pk","wb"))

        #print "Found %s unique tokens" %(len(self.tokenizer.word_index))

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
    
    def make_data_keras(self,fold):
        train_data_k, test_data_k = self.make_data_fold(fold)
                
        bodies_sequence = self.tokenizer.texts_to_sequences(train_data_k[:,0])
        headlines_sequence = self.tokenizer.texts_to_sequences(train_data_k[:,1])

        bodies_sequence_test = self.tokenizer.texts_to_sequences(test_data_k[:,0])
        headlines_sequence_test = self.tokenizer.texts_to_sequences(test_data_k[:,1])
        
        max_seq_length = max(max([len(bodies_sequence[i]) for i in range(len(bodies_sequence))]),\
                                max([len(bodies_sequence_test[i]) for i in range(len(bodies_sequence_test))]))

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
        
        return {"train":[bodies_data, headlines_data, labels],"test":[bodies_data_test, headlines_data_test, labels_test],"max_seq_length":max_seq_length}

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
    pp.preprocess_keras()
    pp.preprocess_stageone()

    
    _ = pp.make_data_keras(0)
    