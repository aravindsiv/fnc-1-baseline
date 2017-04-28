from preprocess_ import PreProcessor

import keras
import keras.backend as K

from keras.models import Sequential
from keras.layers import Embedding, Dense, Input, TimeDistributed, merge, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

files_dict = {"bodies_file":"train_bodies.csv","stances_file":"train_stances.csv",\
             "train_ids":"training_ids.txt","holdout_ids":"hold_out_ids.txt"}
we_file = "glove.6B.300d.txt"
embedding_dim = 300
sentence_hidden_layer_size = 300
activation = 'relu'
RNN = None
dropout_rate = 0.2
batch_size = 256
num_epochs = 50
patience = 5

pp = PreProcessor(files_dict)
pp.tokenize()
data_dict = pp.make_data()
bodies, headlines, labels = data_dict["train"]
bodies_t, headlines_t, labels_t = data_dict["test"]
embedding_matrix = pp.get_embedding_matrix(we_file)

embed = Embedding(len(pp.word_index)+1,embedding_dim,weights=[embedding_matrix],input_length=pp.max_seq_length)

SumEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x,axis=1),output_shape=(sentence_hidden_layer_size,))

translate = TimeDistributed(Dense(sentence_hidden_layer_size,activation=activation))

body = Input(shape=(pp.max_seq_length,))
headline = Input(shape=(pp.max_seq_length,))

bod = embed(body)
head = embed(headline)

bod = translate(bod)
head = translate(head)

rnn = SumEmbeddings # Change this line if you want an RNN. Refer to Smerity's code.
bod = rnn(bod)
head = rnn(head)
bod = BatchNormalization()(bod)
head = BatchNormalization()(head)

joint = merge([bod,head],mode='concat')

joint = Dropout(dropout_rate)(joint)

for i in range(3):
  joint = Dense(2 * sentence_hidden_layer_size,activation=activation)(joint) # No weight regularization being done.
  joint = Dropout(dropout_rate)(joint)
  joint = BatchNormalization()(joint)

pred = Dense(labels.shape[1],activation='softmax')(joint)

model = Model(input = [body,headline],output=pred)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
callbacks = [EarlyStopping(patience=patience), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]

model.fit([bodies, headlines], labels,batch_size=batch_size,epochs=num_epochs,callbacks=callbacks)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([bodies_t, headlines_t],labels_t)
print "Test loss: %s, accuracy: %s" %(loss, acc)

# Save final model, just to be safe
model.save("SumRNN.h5")

