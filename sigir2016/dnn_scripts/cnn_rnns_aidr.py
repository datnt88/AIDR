'''Train CNN+LSTM RNNs on the AIDR tweet classification task.

GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python lstm_rnns_aidr.py

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

# keras related
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Graph, Sequential
from keras.layers.core    import Dense, Dropout, Activation
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.utils.np_utils import accuracy, to_categorical


from utilities import aidr
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics


#other utilities 
import optparse
import logging
import sys



def build_cnn_lstm_rnns(model_type, maxlen, max_features, emb_size=128, emb_matrix=None,
						recur_type='lstm', nb_filter=64, filter_length=3, pool_length=2,
						nb_classes = 2, recur_size=128, dropout_ratio=0.5, tune_emb=True):

	''' run cnn+lstm rnns '''		

	print('Building model:', model_type, 'cnn-lstm')

	#create the emb layer	
	if emb_matrix is not None:
		max_features, emb_size = emb_matrix.shape
		emb_layer = Embedding(max_features, emb_size, weights=[emb_matrix], input_length=maxlen, trainable=tune_emb)
	else:
		emb_layer = Embedding(max_features, emb_size, input_length=maxlen, trainable=tune_emb)

	#create fwd recurrent layer
	if recur_type.lower() == 'lstm':
		recur_layer = LSTM(recur_size)
	elif recur_type.lower() == 'gru': 
		recur_layer = GRU(recur_size)
	elif recur_type.lower() == 'simplernn': 
		recur_layer = SimpleRNN(recur_size)


	if model_type.lower() == 'bidirectional':
		model = Graph()

		# add common input
		model.add_input(name='input', input_shape=(maxlen,), dtype=int) # add word id (input) layer;
		model.add_node(emb_layer,    name='embedding', input='input')   # add the emb node
		model.add_node(Dropout(dropout_ratio),   name='emb_dropout', input='embedding') # add dropout to emb

		# add a cnn layer
		model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, 
                    	border_mode='valid', activation='relu', subsample_length=1), name='conv', input='emb_dropout')
		model.add_node(MaxPooling1D(pool_length=pool_length), name='pool', input='conv')


		#create bwd recurrent layer
		if recur_type.lower() == 'lstm':
			recur_layer2 = LSTM(recur_size, go_backwards=True)
		elif recur_type.lower() == 'gru': 
			recur_layer2 = GRU(recur_size,  go_backwards=True)
		elif recur_type.lower() == 'simplernn': 
			recur_layer2 = SimpleRNN(recur_size, go_backwards=True)

		# add rnns 
		model.add_node(recur_layer,  name='forward',   input='pool')  # fwd lstm layer
		model.add_node(recur_layer2,  name='backward',   input='pool')  # fwd lstm layer

		model.add_node(Dropout(dropout_ratio), name='dropout', inputs=['forward', 'backward'])  # add dropout to lstm layers

		if nb_classes == 2:
			print('Doing binary classification...')
			model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout') # output node
			model.add_output(name='output', input='sigmoid')
		elif nb_classes > 2:
			print('Doing classification with class #', nb_classes)
			model.add_node(Dense(nb_classes, activation='softmax'), name='softmax', input='dropout') # output node
			model.add_output(name='output', input='softmax')

		else:
			print("Wrong argument nb_classes: ", nb_classes)
			exit(1)

	else:
		model = Sequential()
		#emb and dropout
		model.add(emb_layer)	
		model.add(Dropout(dropout_ratio))

		# we add a Convolution1D, which will learn nb_filter (word group) filters of size filter_length:
		model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, 
                        	border_mode='valid', activation='relu', subsample_length=1))

		# we use standard max pooling (halving the output of the previous layer):
		model.add(MaxPooling1D(pool_length=pool_length))
		model.add(recur_layer) 
		model.add(Dropout(dropout_ratio)) 

		if nb_classes == 2:
			print('Doing binary classification...')
			model.add(Dense(1))
			model.add(Activation('sigmoid'))

		elif nb_classes > 2:
			print('Doing classification with class #', nb_classes)
			model.add(Dense(nb_classes))
			model.add(Activation('softmax'))

		else:
			print("Wrong argument nb_classes: ", nb_classes)
			exit(1)

	return model
		






if __name__ == '__main__':


	logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

	# parse user input
	parser = optparse.OptionParser("%prog [options]")

	#file related options
	parser.add_option("-g", "--log-file",          dest="log_file", help="log file [default: %default]")
	parser.add_option("-d", "--data-dir",          dest="data_dir", help="directory containing train, test and dev file [default: %default]")
	parser.add_option("-D", "--data-spec",         dest="data_spec", help="specification for training data (in, out, in_out) [default: %default]")
	parser.add_option("-M", "--model-dir",         dest="model_dir", help="directory to save the best models [default: %default]")

#	parser.add_option("-r", "--train-file",        dest="featFile_train")
#	parser.add_option("-s", "--test-file",         dest="featFile_test")
#	parser.add_option("-v", "--validation-file",   dest="featFile_dev")

	# network related
	parser.add_option("-t", "--max-tweet-length",  dest="maxlen",       type="int", help="maximul tweet length (for fixed size input) [default: %default]") # input size

	# cnn related
	parser.add_option("-F", "--nb_filter",         dest="nb_filter",     type="int",   help="nb of filter to be applied in convolution over words [default: %default]") # uni, bi-directional
	parser.add_option("-k", "--filter_length",     dest="filter_length", type="int",   help="length of neighborhood in words [default: %default]") # lstm, gru, simpleRNN
	parser.add_option("-p", "--pool_length",       dest="pool_length",   type="int",   help="length for max pooling [default: %default]") # lstm, gru, simpleRNN

	# rnn related
	parser.add_option("-m", "--model-type",        dest="model_type",   help="uni or bidirectional [default: %default]") # uni, bi-directional
	parser.add_option("-r", "--recurrent-type",    dest="recur_type",   help="recurrent types (lstm, gru, sipleRNN) [default: %default]") # lstm, gru, simpleRNN

	parser.add_option("-v", "--vocabulary-size",   dest="max_features", type="int", help="vocabulary size [default: %default]") # emb matrix row size
	parser.add_option("-e", "--emb-size",          dest="emb_size",     type="int", help="dimension of embedding [default: %default]") # emb matrix col size
	parser.add_option("-s", "--hidden-size",       dest="hidden_size",   type="int", help="hidden layer size [default: %default]") # size of the hidden layer
	parser.add_option("-o", "--dropout_ratio",     dest="dropout_ratio", type="float", help="ratio of cells to drop out [default: %default]")

	parser.add_option("-i", "--init-type",         dest="init_type",     help="random or pretrained [default: %default]") 
	parser.add_option("-f", "--emb-file",          dest="emb_file",      help="file containing the word vectors [default: %default]") 
	parser.add_option("-P", "--tune-emb",          dest="tune_emb",      action="store_false", help="DON't tune word embeddings [default: %default]") 


	#learning related
	parser.add_option("-a", "--learning-algorithm", dest="learn_alg", help="optimization algorithm (adam, sgd, adagrad, rmsprop, adadelta) [default: %default]")
	parser.add_option("-b", "--minibatch-size",     dest="minibatch_size", type="int", help="minibatch size [default: %default]")
	parser.add_option("-l", "--loss",               dest="loss", help="loss type (hinge, squared_hinge, binary_crossentropy) [default: %default]")
	parser.add_option("-n", "--epochs",             dest="epochs", type="int", help="nb of epochs [default: %default]")


	parser.set_defaults(
    	data_dir       = "../data/earthquakes/in"
    	,data_spec       = "in"
    	,model_dir       = "./saved_models/"
	    ,log_file       = "log"

#    	,featFile_train = "../data/good_vs_bad/CQA-QL-train.xml.multi.csv.feat"
#    	,featFile_test  = "../data/good_vs_bad/CQA-QL-test.xml.multi.csv.feat"
#    	,featFile_dev   = "../data/good_vs_bad/CQA-QL-devel.xml.multi.csv.feat"

	   	,learn_alg      = "adam" # sgd, adagrad, rmsprop, adadelta, adam (default)
	   	,loss           = "binary_crossentropy" # hinge, squared_hinge, binary_crossentropy (default)
	    ,minibatch_size = 20 #32
    	,dropout_ratio  = 0.0

    	,maxlen         = 100
    	,epochs         = 25
    	,max_features   = 10000
    	,emb_size       = 128
    	,hidden_size    = 70 #128 #70
    	,model_type     = 'unidirectional' #bidirectional, unidirectional (default)
    	,recur_type     = 'lstm' #gru, simplernn, lstm (default) 

    	,nb_filter      = 64
    	,filter_length  = 3 
    	,pool_length    = 2 
    	,init_type      = 'random' 
    	,emb_file       = "../data/unlabeled_corpus.vec"
    	,tune_emb       = True
	)

	options,args = parser.parse_args(sys.argv)


	print('Loading data...')
	(X_train, y_train), (X_test, y_test), (X_dev, y_dev), max_features, E, label_id = aidr.load_and_numberize_data(path=options.data_dir,
																			nb_words=options.max_features, init_type=options.init_type,
																			embfile=options.emb_file)

#	print("Padding sequences....")
	X_train = sequence.pad_sequences(X_train, maxlen=options.maxlen)
	X_test  = sequence.pad_sequences(X_test,  maxlen=options.maxlen)
	X_dev   = sequence.pad_sequences(X_dev,   maxlen=options.maxlen)

	#build model...
	nb_classes = np.max(y_train) + 1

	print('............................')
	print(len(X_train), 'train tweets')
	print(len(X_test),  'test  tweets')
	print(len(X_dev),   'dev   tweets')
	print(max_features - 3, 'vocabulary size')
	print(nb_classes, 'different classes')
	print('............................')


	if nb_classes == 2: # binary
		loss       = options.loss
		class_mode = "binary"
		optimizer  = options.learn_alg

	elif nb_classes > 2: # multi-class
		loss       = 'categorical_crossentropy'
		class_mode = 'categorical'
		optimizer  = 'adadelta'

		# convert class vectors to binary class matrices [ 1 of K encoding]
		y_train_mod = to_categorical(y_train, nb_classes)
		y_test_mod  = to_categorical(y_test,  nb_classes)
		y_dev_mod   = to_categorical(y_dev,   nb_classes)


	#build model...
	model = build_cnn_lstm_rnns(options.model_type, options.maxlen, max_features, emb_matrix=E, emb_size=options.emb_size, 
								recur_type=options.recur_type, nb_filter=options.nb_filter, filter_length=options.filter_length,
								pool_length=options.pool_length, nb_classes = nb_classes,
								recur_size=options.hidden_size, dropout_ratio=options.dropout_ratio, tune_emb=options.tune_emb)

	model_name = options.model_dir + options.recur_type + "-cnn-" + options.model_type + "-" + optimizer + "-" + str (options.tune_emb) +\
	"-" + loss + "-" + str (options.minibatch_size) + "-" + str(options.dropout_ratio) + "-" + str(options.nb_filter) + "-init-" + str (options.init_type) +\
	"-" + str (options.max_features) + "-" + str (options.emb_size) + "-" + str (options.hidden_size) + ".model.cl." + str(nb_classes) + ".dom." + str(options.data_spec)


	earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
	checkpointer = ModelCheckpoint(filepath=model_name, monitor='val_loss', verbose=1, save_best_only=True)

	print ('Training and validating ....')

	if options.model_type.lower() != 'bidirectional':
		model.compile(optimizer=optimizer, loss=loss,  class_mode=class_mode)

		if nb_classes == 2: # binary
			model.fit(X_train, y_train, batch_size=options.minibatch_size, nb_epoch=options.epochs,
					validation_data=(X_dev, y_dev), show_accuracy=True, verbose=1, callbacks=[earlystopper, checkpointer])

			print("Test model ...")
			print ("Loading ...", model_name)
			model.load_weights(model_name)

			y_prob = model.predict_proba(X_test)
			y_pred = np.round(y_prob)

			roc = metrics.roc_auc_score(y_test, y_prob)
			print("ROC Prediction (binary classification):", roc)

		elif nb_classes > 2: # multi-class
			model.fit(X_train, y_train_mod, batch_size=options.minibatch_size, nb_epoch=options.epochs,
					validation_data=(X_dev, y_dev_mod), show_accuracy=True, verbose=1, callbacks=[earlystopper, checkpointer])

			print("Test model ...")
			print ("Loading ...", model_name)
			model.load_weights(model_name)
			y_pred = model.predict_classes(X_test)

	else:


		model.compile(optimizer=optimizer, loss={'output':loss})

		if nb_classes == 2: # binary

			model.fit({'input': X_train, 'output': y_train}, batch_size=options.minibatch_size, nb_epoch=options.epochs,
				validation_data={'input': X_dev, 'output': y_dev}, callbacks=[earlystopper, checkpointer])

			print("Test model ...")
			print ("Loading ...", model_name)
			model.load_weights(model_name)

			y_prob = model.predict({'input': X_test}, batch_size=options.minibatch_size)['output']
			y_pred = np.round(y_prob)
 
			roc = metrics.roc_auc_score(y_test, y_prob)
			print("ROC Prediction (binary classification):", roc)

		elif nb_classes > 2: # multi-class

			model.fit({'input': X_train, 'output': y_train_mod}, batch_size=options.minibatch_size, nb_epoch=options.epochs,
				validation_data={'input': X_dev, 'output': y_dev_mod}, callbacks=[earlystopper, checkpointer])

			print("Test model ...")
			print ("Loading ...", model_name)
			model.load_weights(model_name)
			y_prob = model.predict({'input': X_test}, batch_size=options.minibatch_size)['output']
			y_pred = np.argmax(y_prob, axis=1)


	y_test = np.array(y_test)
	
	acc2 = metrics.accuracy_score(y_test, y_pred)
	print("Raw Accuracy:", acc2)

	#get label ids in sorted
	class_labels = sorted(label_id, key=label_id.get)
	class_ids    = sorted(label_id.values())

	print (metrics.classification_report(y_test, y_pred, labels=class_ids, target_names=class_labels) )
	print ("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred) )

	mic_p, mic_r, mic_f, sup = metrics.precision_recall_fscore_support(y_test, y_pred, average='micro')
	mac_p, mac_r, mac_f, sup = metrics.precision_recall_fscore_support(y_test, y_pred, average='macro')

	print (" micro pre: " + str (mic_p) + " rec: " + str (mic_r) + " f-score: " + str (mic_f))
	print (" macro pre: " + str (mac_p) + " rec: " + str (mac_r) + " f-score: " + str (mac_f))

