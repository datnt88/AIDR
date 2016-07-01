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

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop


from utilities import aidr, aidr_feat
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics

#other utilities 
import optparse
import logging
import sys


def build_MLP(model_type, max_features, nb_layers=1, hidden_size=128, nb_classes = 2, dropout_ratio=0.5):

	''' build MLP '''

	print('Building model %s with %d layers', model_type, nb_layers)
	model = Sequential()

	if nb_layers == 0: #no hidden layer
		print('Doing LR classification with class #', nb_classes)
		model.add(Dense(nb_classes, input_dim=max_features)) 
		model.add(Activation('softmax'))

	elif nb_layers > 0: # add one hidden layer
		model.add(Dense(hidden_size, input_shape=(max_features, ))) 
		model.add(Activation('relu'))	
		model.add(Dropout(dropout_ratio))
	
	elif nb_layers > 1: # add another hidden layer
		model.add(Dense(hidden_size))
		model.add(Activation('relu'))
		model.add(Dropout(dropout_ratio))
	else:
		print("layer nb not supported..", nb_layers)
		exit(1)


	# We project onto a single unit output layer, and squash it with a sigmoid:
	if nb_classes == 2 and nb_layers != 0:
		print('Doing binary classification...')
		model.add(Dense(1))
		model.add(Activation('sigmoid'))

	elif nb_classes > 2 and nb_layers != 0:
		print('Doing classification with class #', nb_classes)
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))


	return model	


if __name__ == '__main__':


	logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

	# parse user input
	parser = optparse.OptionParser("%prog [options]")

	#file related options
	parser.add_option("-g", "--log-file",          dest="log_file", help="log file [default: %default]")
	parser.add_option("-d", "--data-dir",          dest="data_dir", help="directory containing train, test and dev file [default: %default]")
	parser.add_option("-D", "--data-spec",         dest="data_spec", help="specification for training data (in, out, in_out) [default: %default]")
	parser.add_option("-p", "--model-dir",          dest="model_dir", help="directory to save the best models [default: %default]")

	# network related
	parser.add_option("-m", "--model-type",        dest="model_type",   help="LR or MLP [default: %default]") # uni, bi-directional
	parser.add_option("-s", "--hidden-size",       dest="hidden_size",   type="int", help="hidden layer size [default: %default]") # size of the hidden layer
	parser.add_option("-o", "--dropout_ratio",     dest="dropout_ratio", type="float", help="ratio of cells to drop out [default: %default]")

	#learning related
	parser.add_option("-a", "--learning-algorithm", dest="learn_alg", help="optimization algorithm (adam, sgd, adagrad, rmsprop, adadelta) [default: %default]")
	parser.add_option("-b", "--minibatch-size",     dest="minibatch_size", type="int", help="minibatch size [default: %default]")
	parser.add_option("-l", "--loss",               dest="loss", help="loss type (hinge, squared_hinge, binary_crossentropy) [default: %default]")
	parser.add_option("-n", "--epochs",             dest="epochs", type="int", help="nb of epochs [default: %default]")
	parser.add_option("-C", "--map-class",          dest="map_class", type="int", help="map classes to five labels [default: %default]")
	parser.add_option("-H", "--nb-layers",          dest="nb_layers", type="int", help="nb of hidden layers [default: %default]")
	parser.add_option("-F", "--add-feature",        dest="add_feat",  type="int", help="nb of hidden layers [default: %default]")

	parser.set_defaults(
    	data_dir        = None
    	,data_spec       = "in"
    	,model_dir       = "./saved_models/"
	    ,log_file       = "log"
	   	,learn_alg      = "adam" # sgd, adagrad, rmsprop, adadelta, adam (default)
	   	,loss           = "hinge"#"binary_crossentropy" # hinge, squared_hinge, binary_crossentropy (default)
	    ,minibatch_size = 32
    	,dropout_ratio  = 0.2
    	,epochs         = 25
    	,hidden_size    = 128
    	,nb_layers      = 1
    	,model_type     = 'mlp' 
    	,add_feat       = 0
    	,map_class      = 0
	)

	options,args = parser.parse_args(sys.argv)

	print('Loading data...')
	(X_train, y_train), (X_test, y_test), (X_dev, y_dev), max_features, E, label_id = aidr.load_and_numberize_data(path=options.data_dir, seed=113)
	X_train_f, X_test_f, X_dev_f = aidr_feat.load_tfidf_vectors(path=options.data_dir, seed=113) # load features

	assert len(X_train) == X_train_f.shape[0] and len(X_test) == X_test_f.shape[0]

	#build model...
	nb_classes = len(label_id)
	max_features = X_train_f.shape[1]
	print('............................')
	print(len(X_train), 'train tweets')
	print(len(X_test),  'test  tweets')
	print(len(X_dev),   'dev   tweets')
	print(max_features, 'features')
	print(nb_classes, 'different classes')
	print('............................')


	if nb_classes == 2: # binary
		loss       = options.loss
		class_mode = "binary"
		optimizer  = options.learn_alg

	elif nb_classes > 2: # multi-class
#		loss       = 'categorical_crossentropy'
		loss       = 'categorical_crossentropy'
		class_mode = 'categorical'
		optimizer  = 'rmsprop' #'adadelta' 
		# convert class vectors to binary class matrices [ 1 of K encoding]
		y_train_mod = to_categorical(y_train, nb_classes)
		y_test_mod  = to_categorical(y_test,  nb_classes)
		y_dev_mod   = to_categorical(y_dev,   nb_classes)
	
	model = build_MLP(options.model_type, max_features, nb_layers=options.nb_layers, hidden_size=options.hidden_size,\
						nb_classes=nb_classes, dropout_ratio=options.dropout_ratio)

	model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

	model_name = options.model_dir + "MLP" + "-layers-" + str(options.nb_layers) + "-hidden-" + str (options.hidden_size) + "-"\
					+ optimizer + "-" + loss + "-" + str (options.minibatch_size) + "-" + str(options.dropout_ratio) + "-"\
					+ str (max_features) + "-" + ".model.cl." + str(nb_classes) + ".dom." + options.data_spec 

	earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
	checkpointer = ModelCheckpoint(filepath=model_name, monitor='val_loss', verbose=1, save_best_only=True)

	if nb_classes == 2: # binary
		print ('Training and validating ....')
		model.fit(X_train_f, y_train, batch_size=options.minibatch_size, nb_epoch=options.epochs,
				validation_data=(X_dev, y_dev), show_accuracy=True, verbose=1, callbacks=[earlystopper, checkpointer])

		print("Test model ...")
		print ("Loading ...", model_name)
		model.load_weights(model_name)

		y_prob = model.predict_proba(X_test)
		roc = metrics.roc_auc_score(y_test, y_prob)
		print("ROC Prediction (binary classification):", roc)


	elif nb_classes > 2: # multi-class
		print ('Training and validating ....')
		model.fit(X_train_f, y_train_mod, batch_size=options.minibatch_size, nb_epoch=options.epochs,
				validation_data=(X_test_f, y_test_mod), verbose=1, callbacks=[earlystopper, checkpointer])

		print("Test model ...")
		print ("Loading ...", model_name)
		model.load_weights(model_name)

	y_pred = model.predict_classes(X_test_f)
	y_test = np.array(y_test)

	
	acc2 = metrics.accuracy_score(y_test, y_pred)
	print("Raw Accuracy:", acc2)

	#get label ids in sorted
	class_labels = sorted(label_id, key=label_id.get)

#	class_labels = map(str, sorted(label_id, key=label_id.get))
	class_ids    = sorted(label_id.values())

	print (metrics.classification_report(y_test, y_pred, labels=class_ids, target_names=class_labels) )

	print ("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred) )

	mic_p, mic_r, mic_f, sup = metrics.precision_recall_fscore_support(y_test, y_pred, average='micro')
	mac_p, mac_r, mac_f, sup = metrics.precision_recall_fscore_support(y_test, y_pred, average='macro')

	print (" micro pre: " + str (mic_p) + " rec: " + str (mic_r) + " f-score: " + str (mic_f))
	print (" macro pre: " + str (mac_p) + " rec: " + str (mac_r) + " f-score: " + str (mac_f))

	# save the architecture finally in json format
	json_string = model.to_json()
	open(model_name + ".json", 'w').write(json_string)


