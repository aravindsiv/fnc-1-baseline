from preprocess import PreProcessor

import keras
import keras.backend as K
import argparse
import tempfile
import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, Dense, Input, TimeDistributed, merge, Dropout, BatchNormalization, recurrent
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2

embedding_dim = 300
sentence_hidden_layer_size = 300
activation = 'relu'
dropout_rate = 0.2
batch_size = 64
num_epochs = 10
patience = 5
L2 = 4e-6

if __name__ == "__main__":
	pp = PreProcessor()
	pp.preprocess_keras()
	embedding_matrix = pp.get_embedding_matrix()

	parser = argparse.ArgumentParser()

	parser.add_argument('-fold',help='fold to train the model on',default='0')
	parser.add_argument('-rnn',help='no | lstm | gru',default='no')

	args = vars(parser.parse_args())

	data_dict = pp.make_data_keras(args['fold'])
	bodies, headlines, labels = data_dict["train"]
	bodies_t, headlines_t, labels_t = data_dict["test"]
	max_seq_length = data_dict["max_seq_length"]

	if args['rnn'] == 'no':
		RNN = None
		fprefix = "SumRNN"
	elif args['rnn'] == 'lstm':
		RNN = recurrent.LSTM 
		fprefix = "lstm"
	elif args ['rnn'] == 'gru':
		RNN = recurrent.GRU
		fprefix = "gru"
	else:
		print "Invalid arg for rnn"
		RNN = None

	embed = Embedding(len(pp.tokenizer.word_index)+1,embedding_dim,weights=[embedding_matrix],input_length=max_seq_length)

	SumEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x,axis=1),output_shape=(sentence_hidden_layer_size,))

	translate = TimeDistributed(Dense(sentence_hidden_layer_size,activation=activation))

	body = Input(shape=(max_seq_length,))
	headline = Input(shape=(max_seq_length,))

	bod = embed(body)
	head = embed(headline)

	bod = translate(bod)
	head = translate(head)

	rnn = SumEmbeddings if not RNN else RNN(return_sequences=False,output_dim=sentence_hidden_layer_size)
	bod = rnn(bod)
	head = rnn(head)
	bod = BatchNormalization()(bod)
	head = BatchNormalization()(head)

	joint = merge([bod,head],mode='concat')

	joint = Dropout(dropout_rate)(joint)

	for i in range(3):
	  joint = Dense(2 * sentence_hidden_layer_size,activation=activation,W_regularizer=l2(L2) if L2 else None)(joint) # No weight regularization being done.
	  joint = Dropout(dropout_rate)(joint)
	  joint = BatchNormalization()(joint)

	pred = Dense(labels.shape[1],activation='softmax')(joint)

	model = Model(input = [body,headline],output=pred)
	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

	model.summary()

	# _, tmpfn = tempfile.mkstemp()
	# Save the best model during validation and bail out of training early if we're not improving
	# callbacks = [EarlyStopping(patience=patience), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]

	per_batch = int(len(bodies)/batch_size)

	indices = np.array(range(len(bodies)))
	np.random.shuffle(indices)
	z = np.array_split(indices,per_batch)

	for j in range(num_epochs):
		for i in range(len(z)):
			loss, accuracy = model.train_on_batch([bodies[z[i]],headlines[z[i]]],labels[z[i]])
			print "Epoch ", j, " Batch ", i, " of ", len(z), " Loss: ", loss, " Accuracy: ", accuracy

	# model.fit([bodies, headlines], labels,batch_size=batch_size,epochs=num_epochs)#,callbacks=callbacks)

	# Restore the best found model during validation
	# model.load_weights(tmpfn)

	loss, acc = model.evaluate([bodies_t, headlines_t],labels_t)
	print "Test loss: %s, accuracy: %s" %(loss, acc)

	# Save final model, just to be safe
	model.save("models/"+fprefix+"_"+str(args['fold'])+".h5")

