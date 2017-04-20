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

def get_stance_matrix(answers, encoder):

	y = encoder.transform(answers) #string to numerical class
	nb_classes = encoder.classes_.shape[0]
	Y = np_utils.to_categorical(y, nb_classes)
	return Y

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
		headline_data[body_id] = headline
		stance_data[body_id] = stance

dict_data = defaultdict(list)

for d in (headline_data, body_data, stance_data):
	for k, v in d.iteritems():
		dict_data[k].append(v)

new_dict_data = {k: v for k,v  in dict_data.iteritems() if v[2]!='unrelated'}

with open('./data/hold_out_ids.txt') as f:
	for line in f:
		if int(line) in new_dict_data:
			hold_out_ids.append(int(line))

with open('./data/training_ids.txt') as f:
	for line in f:
		if int(line) in new_dict_data:
			training_ids.append(int(line))

training_dict = {i:new_dict_data[i] for i in training_ids}
hold_out_dict = {i:new_dict_data[i] for i in hold_out_ids}

headline, body, stance = [item[0] for item in training_dict.values()], [item[1] for item in training_dict.values()], [item[2] for item in training_dict.values()]

headline = [i.decode('utf8') for i in headline]
body = [i.decode('utf8') for i in body]

labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(stance)
nb_classes = len(list(labelencoder.classes_))
print labelencoder

print len(training_ids)
for i in xrange(16):
	for headlines, bodies, stances in zip(grouper(headline, 99), grouper(body, 99), grouper(stance, 99)):
		timesteps_headline = 12 #len(nlp(headlines[-1]))
		timesteps_body = 393 #len(nlp(bodies[-1]))
		headlines_batch = get_timeseries_nlp(headlines, nlp, timesteps_headline)
		bodies_batch = get_timeseries_nlp(bodies, nlp, timesteps_body)
		print headlines_batch.shape, bodies_batch.shape
		x = np.hstack((headlines_batch, bodies_batch))
		# x = x.reshape(1,x.shape[0],x.shape[1])
		y = get_stance_matrix(stances, labelencoder)
		print y.shape
		loss = language_model.train_on_batch(x , y)
		print "Iteration done"