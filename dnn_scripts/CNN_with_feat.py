'''Train LSTM RNNs on the AIDR tweet classification task.

GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python lstm_rnns_aidr.py

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

# keras related
from __future__ import print_function
import numpy as np

np.random.seed(2000)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy, to_categorical

from keras.layers import Input, Convolution1D, Embedding, MaxPooling1D, Dropout, Dense, Flatten, merge

from keras.models import Model

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.optimizers import SGD, Adam, RMSprop


from utilities import aidr, aidr_feat
from sklearn import metrics

#other utilities 
import optparse
import logging
import sys


def build_cnn_mlp(maxlen, vocab_size, max_features, nb_layers=1, E=None, emb_size=128, nb_filter=250, filter_length=3,
			 pool_length=2,	nb_classes = 2,  hidden_size=128, dropout_ratio=0.5, tune_emb=True):

	''' build CNN+MLP '''
	print('Building CNN+MLP model with %d layers', nb_layers)

	# inputs
	input_cnn  = Input(shape=(maxlen, ), dtype='int32', name='input_cnn')
	input_feat = Input(shape=(max_features, ), dtype='float32', name='input_feat')

	# embedding
	if E is not None:
		vocab_size, emb_size = E.shape
		emb_layer = Embedding(output_dim=emb_size, input_dim=vocab_size, weights=[E], input_length=maxlen, trainable=True, dropout=dropout_ratio)
	else:
		emb_layer = Embedding(output_dim=emb_size, input_dim=vocab_size, input_length=maxlen, dropout=dropout_ratio)

	# shared emb layer
	emb_layer_cnn = emb_layer(input_cnn)

	# convolution
	# we add a Convolution1D, which will learn nb_filter (word group) filters of size filter_length:
	conv_layer = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, 
	                    	border_mode='valid', activation='relu', subsample_length=1)
	dropout_conv   = Dropout(dropout_ratio)
	conv_layer_cnn = dropout_conv(conv_layer(emb_layer_cnn))

	# we use standard max pooling (halving the output of the previous layer):
	pool_layer     = MaxPooling1D(pool_length=pool_length)
	dropout_pool   = Dropout(dropout_ratio)
	pool_layer_cnn = dropout_pool(pool_layer(conv_layer_cnn))

	# We flatten the output of the conv layer, so that we can add a vanilla dense layer:
	flat_layer      = Flatten()
	flat_layer_cnn  = flat_layer(pool_layer_cnn)

	x = Dense(hidden_size, activation='relu') ( merge([flat_layer_cnn, input_feat], mode='concat') )

	# output layer
	if nb_classes == 2:
		print('Doing binary classification...')
		main_output = Dense(1, activation='sigmoid', name='main_output')(x)

	elif nb_classes > 2:
		print('Doing classification with class #', nb_classes)
		main_output = Dense(nb_classes, activation='softmax', name='main_output')(x)

	else:
		print("Wrong argument nb_classes: ", nb_classes)
		exit(1)

	model = Model(input=[input_cnn, input_feat], output=[main_output])

	return model	


if __name__ == '__main__':


	logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

	# parse user input
	parser = optparse.OptionParser("%prog [options]")

	#file related options
	parser.add_option("-g", "--log-file",          dest="log_file", help="log file [default: %default]")
	parser.add_option("-d", "--data-dir",          dest="data_dir", help="directory containing train, test and dev file [default: %default]")
	parser.add_option("-D", "--data-spec",         dest="data_spec", help="specification for training data (in, out, in_out) [default: %default]")
	parser.add_option("-M", "--model-dir",          dest="model_dir", help="directory to save the best models [default: %default]")

	# network related
	parser.add_option("-s", "--hidden-size",       dest="hidden_size",   type="int", help="hidden layer size [default: %default]") # size of the hidden layer
	parser.add_option("-o", "--dropout_ratio",     dest="dropout_ratio", type="float", help="ratio of cells to drop out [default: %default]")

	#learning related
	parser.add_option("-a", "--learning-algorithm", dest="learn_alg", help="optimization algorithm (adam, sgd, adagrad, rmsprop, adadelta) [default: %default]")
	parser.add_option("-b", "--minibatch-size",     dest="minibatch_size", type="int", help="minibatch size [default: %default]")
	parser.add_option("-l", "--loss",               dest="loss", help="loss type (hinge, squared_hinge, binary_crossentropy) [default: %default]")
	parser.add_option("-n", "--epochs",             dest="epochs", type="int", help="nb of epochs [default: %default]")
	parser.add_option("-C", "--map-class",          dest="map_class", type="int", help="map classes to five labels [default: %default]")
	parser.add_option("-H", "--nb-layers",          dest="nb_layers", type="int", help="nb of hidden layers [default: %default]")

	# CNN network related
	parser.add_option("-t", "--max-tweet-length",  dest="maxlen",       type="int", help="maximul tweet length (for fixed size input) [default: %default]") # input size
	parser.add_option("-F", "--nb_filter",         dest="nb_filter",     type="int",   help="nb of filter to be applied in convolution over words [default: %default]") # uni, bi-directional
	parser.add_option("-r", "--filter_length",     dest="filter_length", type="int",   help="length of neighborhood in words [default: %default]") # lstm, gru, simpleRNN
	parser.add_option("-p", "--pool_length",       dest="pool_length",   type="int",   help="length for max pooling [default: %default]") # lstm, gru, simpleRNN
	parser.add_option("-v", "--vocabulary-size",   dest="max_features",  type="float",   help="vocabulary size in percentage [default: %default]") # emb matrix row size
	parser.add_option("-e", "--emb-size",          dest="emb_size",      type="int",   help="dimension of embedding [default: %default]") # emb matrix col size
	parser.add_option("-i", "--init-type",         dest="init_type",     help="random or pretrained [default: %default]") 
	parser.add_option("-f", "--emb-file",          dest="emb_file",      help="file containing the word vectors [default: %default]") 
	parser.add_option("-P", "--tune-emb",          dest="tune_emb",      action="store_false", help="DON't tune word embeddings [default: %default]") 

	parser.set_defaults(
    	data_dir        = None
    	,data_spec       = "in"
    	,model_dir       = "./saved_models/"
	    ,log_file       = "log"
	   	,learn_alg      = "rmsprop" # sgd, adagrad, rmsprop, adadelta, adam (default)
	   	,loss           = "binary_crossentropy" # hinge, squared_hinge, binary_crossentropy (default)
	    ,minibatch_size = 32
    	,dropout_ratio  = 0.2
    	,epochs         = 25
    	,emb_size       = 128
    	,hidden_size    = 128
    	,nb_layers      = 1
    	,nb_filter      = 250
    	,init_type      = 'random' 
    	,filter_length  = 3 
    	,pool_length    = 2 
    	,add_feat       = 0
    	,maxlen         = 80
    	,tune_emb       = True
    	,max_features   = 80
    	,map_class      = 0
	)

	options,args = parser.parse_args(sys.argv)

	print('Loading data...')
	(X_train, y_train), (X_test, y_test), (X_dev, y_dev), vocab_size, E, label_id = aidr.load_and_numberize_data(path=options.data_dir, seed=113,
																			nb_words=options.max_features, init_type=options.init_type,
																			embfile=options.emb_file)
	X_train_f, X_test_f, X_dev_f = aidr_feat.load_tfidf_vectors(path=options.data_dir, seed=113) # load features

	assert len(X_train) == X_train_f.shape[0] and len(X_test) == X_test_f.shape[0]

	print("Padding sequences....")
	X_train = sequence.pad_sequences(X_train, maxlen=options.maxlen)
	X_test  = sequence.pad_sequences(X_test,  maxlen=options.maxlen)
	X_dev   = sequence.pad_sequences(X_dev,   maxlen=options.maxlen)


	#build model...
	nb_classes = len(label_id)
	max_features = X_train_f.shape[1]

	print('............................')
	print(len(X_train), 'train tweets')
	print(len(X_test),  'test  tweets')
	print(len(X_dev),   'dev   tweets')
	print(max_features, 'features')
	print(vocab_size - 3, 'vocabulary size')
	print(nb_classes, 'different classes')
	print('............................')


	if nb_classes == 2: # binary
		loss       = options.loss
#		class_mode = "binary"
		optimizer  = options.learn_alg

	elif nb_classes > 2: # multi-class
		loss       = 'categorical_crossentropy'
#		class_mode = 'categorical'
		optimizer  = 'rmsprop'
		# convert class vectors to binary class matrices [ 1 of K encoding]
		y_train_mod = to_categorical(y_train, nb_classes)
		y_test_mod  = to_categorical(y_test,  nb_classes)
		y_dev_mod   = to_categorical(y_dev,   nb_classes)


	model = build_cnn_mlp(options.maxlen, vocab_size, max_features, nb_layers=options.nb_layers, E=E, emb_size=options.emb_size,\
						nb_filter=options.nb_filter,filter_length=options.filter_length, pool_length=options.pool_length, nb_classes=nb_classes,\
						hidden_size=options.hidden_size, dropout_ratio=options.dropout_ratio, tune_emb=options.tune_emb)

	model.compile(optimizer=optimizer, loss=loss, loss_weights={'main_output': 1.}, metrics=['accuracy'])

	model_name = options.model_dir + "cnn-mlp" + "-" + optimizer + "-" + str(options.nb_filter) + "-" + str (options.tune_emb) +\
	"-" + loss + "-" + str (options.minibatch_size) + "-" + str(options.dropout_ratio) + "-init-" + str (options.init_type) + "-" +\
	str (options.max_features) + "-" + str (options.emb_size) + "-" + str (options.hidden_size) + ".model.cl." + str(nb_classes) + ".dom." + str(options.data_spec) 

	earlystopper = EarlyStopping(monitor='val_acc', patience=5, verbose=1)
	checkpointer = ModelCheckpoint(filepath=model_name, monitor='val_acc', verbose=1, save_best_only=True)

	print ('Training and validating ....')

	if nb_classes > 2:
		model.fit({'input_cnn': X_train, 'input_feat': X_train_f},
	              {'main_output': y_train_mod},
	              validation_data=([X_dev, X_dev_f], y_dev_mod),
	              callbacks=[earlystopper, checkpointer],
	              nb_epoch=options.epochs, batch_size=options.minibatch_size)

		print("Test model ...")
		print ("Loading ...", model_name)
		model.load_weights(model_name)
		y_prob = model.predict([X_test, X_test_f], batch_size=1000)

	else:
		print ('binary not ready yet...' )

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

	# save the architecture finally in json format
#	json_string = model.to_json()
#	open(model_name + ".json", 'w').write(json_string)
