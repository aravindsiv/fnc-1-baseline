import argparse
import sys

from preprocess import PreProcessor
from keras_models import test_model
from keras.layers import recurrent

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-fold', help='fold to train the model on', default='0')
    parser.add_argument('-rnn', help='no | lstm | lstm_1 | gru', default='no')

    args = vars(parser.parse_args())

    k_fold = args['fold']

    if args['rnn'] == 'no':
        RNN = None
        fprefix = "SumRNN"
    elif args['rnn'] == 'lstm':
        RNN = recurrent.LSTM
        fprefix = "lstm"
    elif args['rnn'] == 'gru':
        RNN = recurrent.GRU
        fprefix = "gru"
    elif args['rnn'] == 'lstm_1':
        RNN = recurrent.LSTM
    else:
        print "Invalid arg for rnn"
        RNN = None
        sys.exit(0)

    pp = PreProcessor()
    pp.preprocess_keras()
    pp.preprocess_stageone()

    data_dict = pp.make_data_keras(k_fold)

    model_file = "models/" + fprefix + "_" + str(k_fold) + ".h5"
    test_model(model_file, data_dict)
