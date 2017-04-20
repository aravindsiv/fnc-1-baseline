import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from spacy.en import English
import numpy as np
import csv
from collections import defaultdict
from itertools import izip_longest
from keras.utils import np_utils
from sklearn import preprocessing
import sys

labels = {'agree':0, 'disagree':1, 'discuss':2}

def get_stance_matrix(answers, encoder):
	y = []
	for i in answers:
		y.append(labels[i])
	return np.array(y)
	# y = encoder.transform(answers) #string to numerical class
	# nb_classes = encoder.classes_.shape[0]
	# Y = np_utils.to_categorical(y, nb_classes)
	# return Y

def get_timeseries_nlp(text, nlp, timesteps):
	nb_samples = len(text)
	word_vec_dim = 300
	tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
	for i in xrange(len(text)):
		tokens = nlp(text[i])
		for j in xrange(len(tokens)):
			if j < timesteps:
				tensor[i,j,:] += tokens[j].vector
	return tensor

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

language_model = Sequential()

hypo_dim = 300
prem_dim = 300
max_len = 405
word_vec_dim = 300
nlp = English()

language_model.add(LSTM(200,input_shape=(max_len, word_vec_dim), return_sequences=True))
language_model.add(LSTM(200,return_sequences=True))
language_model.add(LSTM(200, return_sequences=False))
language_model.add(Dense(3, activation='softmax'))

language_model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
language_model.summary()

headline_data = {}
body_data = {}
stance_data = {}

hold_out_ids = []
training_ids = []


with open('./data/train_bodies.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		body_id=int(row[0])
		body = row[1]
		body_data[body_id] = body

with open('./data/train_stances.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		headline = row[0]
		body_id = int(row[1])
		stance = row[2]
		headline_data.setdefault(body_id,[])
		stance_data.setdefault(body_id,[])
		headline_data[body_id].append(headline)
		stance_data[body_id].append(stance)


with open('./data/hold_out_ids.txt') as f:
	for line in f:
		hold_out_ids.append(int(line))

with open('./data/training_ids.txt') as f:
	for line in f:
		training_ids.append(int(line))

training_dict_data = []

holdout_dict_data = []

for d in body_data:
	for j in range(len(headline_data[d])):
		if (stance_data[d][j]!='unrelated') and d in training_ids:
			training_dict_data.append([body_data[d].decode('utf8'),headline_data[d][j].decode('utf8'),stance_data[d][j]])
		elif (stance_data[d][j]!='unrelated') and d in hold_out_ids:
			holdout_dict_data.append([body_data[d].decode('utf8'),headline_data[d][j].decode('utf8'),stance_data[d][j]])


training_dict_data = np.array(training_dict_data)
holdout_dict_data = np.array(holdout_dict_data)

training_dict_data[:,2] = np.array([labels[label] for label in training_dict_data[:,2]])

batch_size = 100

batch_ids = np.arange(0,len(training_dict_data),batch_size)
print len(training_dict_data)
print batch_ids

indices = np.linspace(0,len(training_dict_data)-1,dtype='int')

for i in range(0, len(indices)-1):
	start = indices[i]
	end = indices[i+1]
	timesteps_headline = 12
	timesteps_body = 393
	batch = training_dict_data[start:end]
	bodies = batch[:,0]
	headlines = batch[:,1]
	stances = batch[:,2]

	headlines_batch = get_timeseries_nlp(headlines, nlp, timesteps_headline)
	bodies_batch = get_timeseries_nlp(bodies, nlp, timesteps_body)

	x = np.hstack((headlines_batch, bodies_batch))
	y = get_stance_matrix(stance)

	loss = language_model.train_on_batch(x,y)
	print "Iteration Done"
	break

# print len(training_ids)
# for i in xrange(16):
# 	for headlines, bodies, stances in training_dict:
# 		timesteps_headline = 12 #len(nlp(headlines[-1]))
# 		timesteps_body = 393 #len(nlp(bodies[-1]))
# 		headlines_batch = get_timeseries_nlp(headlines, nlp, timesteps_headline)
# 		bodies_batch = get_timeseries_nlp(bodies, nlp, timesteps_body)
# 		print headlines_batch.shape, bodies_batch.shape
# 		x = np.hstack((headlines_batch, bodies_batch))
# 		# x = x.reshape(1,x.shape[0],x.shape[1])
# 		y = get_stance_matrix(stances, labelencoder)
# 		print y.shape
# 		loss = language_model.train_on_batch(x , y)
# 		print "Iteration done"