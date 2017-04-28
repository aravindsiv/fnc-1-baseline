'''
TODO: 
1. Instead of reading the hold_out_ids and training_ids from file, we should try to generate them each time we train, 
for robustness.
'''

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from collections import defaultdict

import csv
import numpy as np
import unicodedata

def _normalize(text):
    return unicodedata.normalize('NFKD', text.decode('utf8')).encode('ascii', 'ignore')
    

class PreProcessor:
    def __init__(self,files_dict):
        bodies = {}
        headlines = defaultdict(list)
        stances = defaultdict(list)
        
        train_ids = []
        holdout_ids = []

        with open(files_dict["bodies_file"]) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                bodies[int(row[0])] = row[1]
                
        with open(files_dict["stances_file"]) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                headlines[int(row[1])].append(row[0])
                stances[int(row[1])].append(row[2])
                
        with open(files_dict["train_ids"]) as f:
            for l in f:
                train_ids.append(int(l))
        
        with open(files_dict["holdout_ids"]) as f:
            for l in f:
                holdout_ids.append(int(l))
                
        self.train_data = []
        self.holdout_data = []
        
        for i in bodies:
            for j in range(len(headlines[i])):
                if stances[i][j] != "unrelated":
                    if i in train_ids:
                        self.train_data.append([_normalize(bodies[i]),_normalize(headlines[i][j]),_normalize(stances[i][j])])
                    else:
                        self.holdout_data.append([_normalize(bodies[i]),_normalize(headlines[i][j]),_normalize(stances[i][j])])
                        
        self.train_data = np.array(self.train_data)
        self.holdout_data = np.array(self.holdout_data)
        
        print "Number of training examples: %s" %(len(self.train_data))
        print "Number of hold out examples: %s" %(len(self.holdout_data))
    
    def tokenize(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(np.vstack([self.train_data[:,0:2],self.holdout_data[:,0:2]]).flatten())
                
        self.bodies_sequence = tokenizer.texts_to_sequences(self.train_data[:,0])
        self.headlines_sequence = tokenizer.texts_to_sequences(self.train_data[:,1])

        self.bodies_sequence_test = tokenizer.texts_to_sequences(self.holdout_data[:,0])
        self.headlines_sequence_test = tokenizer.texts_to_sequences(self.holdout_data[:,1])
        
        self.word_index = tokenizer.word_index
        
        print "Found %s unique tokens" %(len(tokenizer.word_index))
    
    def make_data(self):
        self.max_seq_length = max(max([len(self.bodies_sequence[i]) for i in range(len(self.bodies_sequence))]),\
        						max([len(self.bodies_sequence_test[i]) for i in range(len(self.bodies_sequence_test))]))
        
        bodies_data = pad_sequences(self.bodies_sequence,maxlen=self.max_seq_length)
        headlines_data = pad_sequences(self.headlines_sequence,maxlen=self.max_seq_length)

        bodies_data_test = pad_sequences(self.bodies_sequence_test,maxlen=self.max_seq_length)
        headlines_data_test = pad_sequences(self.headlines_sequence_test,maxlen=self.max_seq_length)
        
        stances = set(self.train_data[:,2])
        stances_test = set(self.holdout_data[:,2])
        
        label_index = {} # labels_index["agree"]
        stances_index = {} # stances_index[0]
        
        for i,j in enumerate(stances):
            label_index[j] = i
            stances_index[i] = j
        
        labels = np.zeros((len(self.train_data),1))
        labels_test = np.zeros((len(self.holdout_data),1))
        
        for i in range(len(self.train_data)):
            labels[i] = label_index[self.train_data[i,2]]

        for i in range(len(self.holdout_data)):
        	labels_test[i] = labels_index[self.holdout_data[i,2]]

        labels = to_categorical(labels)
        labels_test = to_categorical(labels_test)
        
        print "Shape ofs bodies data tensor: " +str(bodies_data.shape)
        print "Shape of headlines data tensor: " +str(headlines_data.shape)
        print "Shape of labels tensor: " + str(labels.shape)
        
        return {"train":[bodies_data, headlines_data, labels],"test":[bodies_data_test, headlines_data_test, labels_test]}

    def get_embedding_matrix(self,we_file):
        embeddings_index = {}
        
        f = open(we_file)
        
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        
        print "Found %s word vectors." % len(embeddings_index)
            
        self.embedding_matrix = np.zeros((len(self.word_index)+1, 300)) # Change this

        
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

        return self.embedding_matrix
