from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np

import glob, os, csv, re
from collections import Counter




def load_data(path="../data/", map_labels_to_five_class=0, add_feat=0, dev_train_merge=0, seed=113):

    """ get the training, test and dev data """

    S_train, S_test, S_dev = [], [], []
    L_train, L_test, L_dev = [], [], []
    P_train, P_test, P_dev = [], [], []
    U_train, U_test, U_dev = [], [], []

    y_train, y_test, y_dev = [], [], []

    for filename in glob.glob(os.path.join(path, '*.csv')):

        print "Reading data from" + filename
        reader  = csv.reader(open(filename, 'rb'))

        for rowid, row in enumerate (reader):
            if rowid == 0: #header
                continue

            if re.search("train", filename.lower()):    
                get_feat(row[3], S_train, L_train, P_train, U_train)
                y_train.append(row[2])    

            elif re.search("test", filename.lower()):    
                get_feat(row[3], S_test, L_test, P_test, U_test)    
                y_test.append(row[2])    

            elif re.search("dev", filename.lower()):    
                get_feat(row[3], S_dev, L_dev, P_dev, U_dev)    
                y_dev.append(row[2])    

    print "Nb of sentences: train: " + str (len(S_train)) + " test: " + str (len(S_test)) + " dev: " + str (len(S_dev))

    #binarize speaker and unigram features
    nb_speaker   = max(max(S_train), max(S_test), max(S_dev)) + 1

    S_train_bin  = one_of_k(S_train, nb_speaker)
    S_test_bin   = one_of_k(S_test, nb_speaker)
    S_dev_bin    = one_of_k(S_dev, nb_speaker)

    allUnig = [item for sublist in (U_train + U_test + U_dev) for item in sublist]
    nb_items = max(allUnig)+1

    U_train_bin = one_of_k_unig(U_train, nb_items)
    U_dev_bin   = one_of_k_unig(U_dev, nb_items)
    U_test_bin  = one_of_k_unig(U_test, nb_items)


    L_train = np.array(L_train).reshape(len(L_train), 1)
    L_test  = np.array(L_test).reshape(len(L_test), 1)
    L_dev   = np.array(L_dev).reshape(len(L_dev), 1)

    P_train = np.array(P_train).reshape(len(P_train), 1)
    P_test  = np.array(P_test).reshape(len(P_test), 1)
    P_dev   = np.array(P_dev).reshape(len(P_dev), 1)

    if add_feat:
        X_train = np.concatenate((L_train, P_train, S_train_bin, U_train_bin), axis=1)
        X_test  = np.concatenate((L_test,  P_test,  S_test_bin,  U_test_bin),  axis=1)
        X_dev   = np.concatenate((L_dev,   P_dev,   S_dev_bin,   U_dev_bin),   axis=1)
    else:    
        X_train = U_train_bin
        X_test  = U_test_bin
        X_dev   = U_dev_bin

    #-----------------------------------------------------------------------------

    #Create label dictionary that map label to ID
    merge_labels = None

    if map_labels_to_five_class:                

        merge_labels = {
                    # QUESTION/REQUEST
                    "QH":"Ques",\
                    "QO":"Ques",\
                    "QR":"Ques",\
                    "QW":"Ques",\
                    "QY":"Ques",\
                    # APPRECIATION/ASSESSMENT/POLITE 
                    "AA":"Polite",\
                    "P":"Polite",\
                    # STATEMENT 
                    "S":"St",\
                    # RESPONSE 
                    "A":"Res",\
                    "R":"Res",\
                    "U":"Res",\
                    #SUGGESTION
                    "AC":"Sug"}

        y_train = remap_labels(y_train, merge_labels=merge_labels)
        y_test  = remap_labels(y_test,  merge_labels=merge_labels)
        y_dev   = remap_labels(y_dev,   merge_labels=merge_labels)


    label_list = list (set(y_train))
    label_map  = {}
    for lab_id, lab in enumerate (label_list):
        label_map[lab] = lab_id


    # Numberize the labels
    (y_train, y_train_freq)   = numberize_labels(y_train, label_map)
    (y_test,  y_test_freq)    = numberize_labels(y_test,  label_map)
    (y_dev,   y_dev_freq)     = numberize_labels(y_dev,   label_map)

    assert len(X_train) == len(y_train) or len(X_test) == len(y_test) or len(X_dev) == len(y_dev)

    #randomly shuffle the training data
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    if dev_train_merge:
        X_train.extend(X_dev)
        y_train.extend(y_dev)

    return (X_train, y_train), (X_test, y_test), (X_dev, y_dev), label_map

def load_tfidf_vectors(path="../data/", sub="*feat.npy", dev_train_merge=0, seed=113):

    print "Loading tf*idf sparse features "

    train_feat, test_feat, dev_feat = None, None, None

    for filename in glob.glob(os.path.join(path, sub)):

        if re.search("train", filename.lower()): 
            print "Loading train features from " + filename   
            train_feat = np.load(filename)

        if re.search("test", filename.lower()):    
            print "Loading test features from " + filename   
            test_feat = np.load(filename)

        if re.search("dev", filename.lower()):    
            print "Loading dev features from " + filename   
            dev_feat = np.load(filename)

    #randomly shuffle the training data
    np.random.seed(seed)
    np.random.shuffle(train_feat)

    return (train_feat, test_feat, dev_feat)        





def load_data_unig(path="../data/", map_labels_to_five_class=0, add_feat=0, dev_train_merge=0, seed=113):

    """ get the training, test and dev data """

    U_train, U_test, U_dev = [], [], []
    y_train, y_test, y_dev = [], [], []

    for filename in glob.glob(os.path.join(path, '*unigram.compress.csv')):

        print "Reading data from" + filename
        reader  = csv.reader(open(filename, 'rb'))

        for rowid, row in enumerate (reader):
            if rowid == 0: #header
                continue

            if re.search("train", filename.lower()):    
                get_feat_unig(row[3], U_train)
                y_train.append(row[2])    

            elif re.search("test", filename.lower()):    
                get_feat_unig(row[3], U_test)    
                y_test.append(row[2])    

            elif re.search("dev", filename.lower()):    
                get_feat_unig(row[3], U_dev)    
                y_dev.append(row[2])    

    print "Nb of sentences: train: " + str (len(U_train)) + " test: " + str (len(U_test)) + " dev: " + str (len(U_dev))

    #binarize sunigram features
    allUnig = [item for sublist in (U_train + U_test + U_dev) for item in sublist]
    nb_items = max(allUnig)+1

    print nb_items

    X_train = one_of_k_unig(U_train, nb_items)
    X_dev   = one_of_k_unig(U_dev, nb_items)
    X_test  = one_of_k_unig(U_test, nb_items)

    #-----------------------------------------------------------------------------

    #Create label dictionary that map label to ID
    merge_labels = None

    if map_labels_to_five_class:                
        merge_labels = {
                    # QUESTION/REQUEST
                    "QH":"Ques",\
                    "QO":"Ques",\
                    "QR":"Ques",\
                    "QW":"Ques",\
                    "QY":"Ques",\
                    "ques":"Ques",\
                    # APPRECIATION/ASSESSMENT/POLITE 
                    "AA":"Polite",\
                    "P":"Polite",\
                    "appr":"Polite",\
                    # STATEMENT 
                    "S":"St",\
                    "st":"St",\
                    # RESPONSE 
                    "A":"Res",\
                    "R":"Res",\
                    "U":"Res",\
                    "res":"Res",\
                    #SUGGESTION
                    "sug":"Sug",\
                    "AC":"Sug"}

        y_train = remap_labels(y_train, merge_labels=merge_labels)
        y_test  = remap_labels(y_test,  merge_labels=merge_labels)
        y_dev   = remap_labels(y_dev,   merge_labels=merge_labels)


    label_list = list (set(y_train))
    label_map  = {}
    for lab_id, lab in enumerate (label_list):
        label_map[lab] = lab_id

    print label_map

    # Numberize the labels
    (y_train, y_train_freq)   = numberize_labels(y_train, label_map)
    (y_test,  y_test_freq)    = numberize_labels(y_test,  label_map)
    (y_dev,   y_dev_freq)     = numberize_labels(y_dev,   label_map)

    assert len(X_train) == len(y_train) or len(X_test) == len(y_test) or len(X_dev) == len(y_dev)

    #randomly shuffle the training data
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    if dev_train_merge:
        X_train.extend(X_dev)
        y_train.extend(y_dev)

    return (X_train, y_train), (X_test, y_test), (X_dev, y_dev), label_map

def load_data_emb_file(path="../data/", dev_train_merge=0, seed=113):

    """ get the training, test and dev data """

    X_train, X_test, X_dev = [], [], []
    y_train, y_test, y_dev = [], [], []

    for filename in glob.glob(os.path.join(path, '*.emb.csv')):
        print "Reading data from" + filename
        reader  = csv.reader(open(filename, 'rb'))

        for rowid, row in enumerate (reader):
            if re.search("train", filename.lower()):
                X_train.append(map(float, row[2:]))    
                y_train.append(row[1])    

            elif re.search("test", filename.lower()):    
                X_test.append(map(float, row[2:]))    
                y_test.append(row[1])    

            elif re.search("dev", filename.lower()):
                X_dev.append(map(float, row[2:]))    
                y_dev.append(row[1])    

    print "Nb of sentences: train: " + str (len(X_train)) + " test: " + str (len(X_test)) + " dev: " + str (len(X_dev))

    #-----------------------------------------------------------------------------

    label_list = list (set(y_train))

    label_map  = {}
    for lab_id, lab in enumerate (label_list):
        label_map[lab] = lab_id

    # Numberize the labels
    (y_train, y_train_freq)   = numberize_labels(y_train, label_map)
    (y_test,  y_test_freq)    = numberize_labels(y_test,  label_map)
    (y_dev,   y_dev_freq)     = numberize_labels(y_dev,   label_map)

    assert len(X_train) == len(y_train) or len(X_test) == len(y_test) or len(X_dev) == len(y_dev)

    #randomly shuffle the training data
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    if dev_train_merge:
        X_train.extend(X_dev)
        y_train.extend(y_dev)

    X_train = np.array(X_train)    
    y_train = np.array(y_train)    
    X_test  = np.array(X_test)    
    y_test = np.array(y_test)    
    X_dev = np.array(X_dev)    
    y_dev = np.array(y_dev)    

    return (X_train, y_train), (X_test, y_test), (X_dev, y_dev), label_map


def one_of_k_unig(X, nb_items):

    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''

    X_bin = np.zeros((len(X), nb_items))
#    np.set_printoptions(threshold=np.nan)

    for sid, asen in enumerate(X):
        for wid in asen:
            X_bin[sid, wid] = 1.

    return X_bin        




def one_of_k(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    y = np.asarray(y, dtype='int32')

    if not nb_classes:
        nb_classes = np.max(y)+1

    Y = np.zeros((len(y), nb_classes))

    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def get_feat(line, S, L, P, U):
    
    all_feats = line.split()

    S.append( int (all_feats[0].split(":")[1])) #get spaker
    L.append( int (all_feats[1].split(":")[1])) #get length
    P.append( float (all_feats[2].split(":")[1])) #get position
    U.append([int (afeat.split(":")[1]) for afeat in all_feats[3:]]) # get unigrams

def get_feat_unig(line, U):
    all_feats = line.split()
    U.append([int (afeat.split(":")[1]) for afeat in all_feats[0:]]) # get unigrams

def load_and_numberize_data(path="../data/", nb_words=None, maxlen=None, seed=113,\
                            start_char=1, oov_char=2, index_from=3, init_type="random",\
                            embfile=None, dev_train_merge=0, map_labels_to_five_class=0,\
                            out_model=None, out_vocab_file=None):

    """ numberize the train, dev and test files """

    # read the vocab from the entire corpus (train + test + dev)
    vocab = Counter()

    sentences_train = []
    y_train = []

    sentences_test  = []
    y_test = []

    sentences_dev   = []
    y_dev  = []

    for filename in glob.glob(os.path.join(path, '*.csv')):
#        print "Reading vocabulary from" + filename
        reader  = csv.reader(open(filename, 'rb'))

        for rowid, row in enumerate (reader):
            if rowid == 0: #header
                continue
            if re.search("train", filename.lower()):    
                sentences_train.append(row[1])
                y_train.append(row[2])    

            elif re.search("test", filename.lower()):    
                sentences_test.append(row[1])    
                y_test.append(row[2])    

            elif re.search("dev", filename.lower()):    
                sentences_dev.append(row[1])    
                y_dev.append(row[2])    

            for wrd in row[1].split():
                vocab[wrd] += 1


    full_vocab_size = len(vocab)
    full_vocab      = dict (vocab) 

    print "Nb of sentences: train: " + str (len(sentences_train)) + " test: " + str (len(sentences_test)) + " dev: " + str (len(sentences_dev))
    print "Total vocabulary size: " + str (full_vocab_size)

    if nb_words is None: # now take a fraction
        nb_words = len(vocab) 
    else:
        pr_perc  = nb_words
        nb_words = int ( len(vocab) * (nb_words / 100.0) )    

#    if nb_words is None or nb_words > len(vocab):
#        nb_words = len(vocab) 

    vocab = dict (vocab.most_common(nb_words))

    print "Pruned vocabulary size: " + str (pr_perc) + "% =" + str (len(vocab))

    #Create vocab dictionary that maps word to ID
    vocab_list = vocab.keys()
    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    # Numberize the sentences
    X_train = numberize_sentences(sentences_train, vocab_idmap, oov_char=oov_char)
    X_test  = numberize_sentences(sentences_test,  vocab_idmap, oov_char=oov_char)
    X_dev   = numberize_sentences(sentences_dev,   vocab_idmap, oov_char=oov_char)


    #Create label dictionary that map label to ID
    merge_labels = None

    if map_labels_to_five_class:                

        merge_labels = {
                    # QUESTION/REQUEST
                    "QH":"Ques",\
                    "QO":"Ques",\
                    "QR":"Ques",\
                    "QW":"Ques",\
                    "QY":"Ques",\
                    # APPRECIATION/ASSESSMENT/POLITE 
                    "AA":"Polite",\
                    "P":"Polite",\
                    # STATEMENT 
                    "S":"St",\
                    # RESPONSE 
                    "A":"Res",\
                    "R":"Res",\
                    "U":"Res",\
                    #SUGGESTION
                    "AC":"Sug"}

        y_train = remap_labels(y_train, merge_labels=merge_labels)
        y_test  = remap_labels(y_test,  merge_labels=merge_labels)
        y_dev   = remap_labels(y_dev,   merge_labels=merge_labels)

    label_list = list (set(y_train))
#    print "labels:", label_list
    label_map  = {}
    for lab_id, lab in enumerate (label_list):
        label_map[lab] = lab_id  

    # Numberize the labels
    (y_train, y_train_freq)   = numberize_labels(y_train, label_map)
    (y_test,  y_test_freq)    = numberize_labels(y_test,  label_map)
    (y_dev,   y_dev_freq)     = numberize_labels(y_dev,   label_map)


    assert len(X_train) == len(y_train) or len(X_test) == len(y_test) or len(X_dev) == len(y_dev)

    #randomly shuffle the training data
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)


    X_train, y_train = adjust_index(X_train, y_train, start_char=start_char, index_from=index_from, maxlen=maxlen)
    X_test,  y_test  = adjust_index(X_test,  y_test,  start_char=start_char, index_from=index_from, maxlen=maxlen)
    X_dev,   y_dev   = adjust_index(X_dev,   y_dev,   start_char=start_char, index_from=index_from, maxlen=maxlen)

    if dev_train_merge:
        X_train.extend(X_dev)
        y_train.extend(y_dev)
#        y_train=np.concatenate ((y_train, y_dev)) # need if y_train is numpy array
 
    # load the embeddeings
    if init_type.lower() != "random" and embfile:
        E = load_emb(embfile, vocab_idmap, index_from)
    elif init_type.lower() != "random" and out_model:
        E = get_embeddings_from_weight_file(out_model, vocab_idmap, out_vocab_file, index_from)
    else:
        E = None


    # prepare dummy features
    

    return (X_train, y_train), (X_test, y_test), (X_dev, y_dev), nb_words + index_from, E, label_map


def extract_and_wrtie_vocab(datadir, nb_words, vocab_file):

    """ numberize the train, dev and test files """

    # read the vocab from the entire corpus (train + test + dev)
    vocab = Counter()

    sentences_train = []
    y_train = []

    sentences_test  = []
    y_test = []

    sentences_dev   = []
    y_dev  = []

    for filename in glob.glob(os.path.join(datadir, '*.csv')):
#        print "Reading vocabulary from" + filename
        reader  = csv.reader(open(filename, 'rb'))

        for rowid, row in enumerate (reader):
            if rowid == 0: #header
                continue
            if re.search("train", filename.lower()):    
                sentences_train.append(row[1])
                y_train.append(row[2])    

            elif re.search("test", filename.lower()):    
                sentences_test.append(row[1])    
                y_test.append(row[2])    

            elif re.search("dev", filename.lower()):    
                sentences_dev.append(row[1])    
                y_dev.append(row[2])    

            for wrd in row[1].split():
                vocab[wrd] += 1

    print "Nb of sentences: train: " + str (len(sentences_train)) + " test: " + str (len(sentences_test)) + " dev: " + str (len(sentences_dev))
    print "Total vocabulary size: " + str (len(vocab))

    if nb_words is None: # now take a fraction
        nb_words = len(vocab) 
    else:
        pr_perc  = nb_words
        nb_words = int ( len(vocab) * (nb_words / 100.0) )    

    vocab = dict (vocab.most_common(nb_words))
    print "Pruned vocabulary size: " + str (pr_perc) + "% =" + str (len(vocab))

    #Create vocab dictionary that maps word to ID
    print "Saving vocab to " + vocab_file

    vocab_list = vocab.keys()
    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    cPickle.dump(vocab_idmap, open(vocab_file, "wb"))    

#    fh = open(vocab_file, "w")
#    for word in sorted(vocab_idmap, key=vocab_idmap.get):
#        print>>fh, word + "\t" + str (vocab_idmap[word]) 
#    fh.close()    

def load_emb(embfile, vocab_idmap, index_from=3, start_char=1, oov_char=2, padding_char=0, vec_size=300):
    """ load the word embeddings """

    print "Loading pre-trained word2vec embeddings......"

    if embfile.endswith(".gz"):
        f = gzip.open(embfile, 'rb')
    else:
        f = open(embfile, 'rb')

    vec_size_got = int ( f.readline().strip().split()[1]) # read the header to get vec dim

    if vec_size_got != vec_size:
        print " vector size provided and found in the file don't match!!!"
        raw_input(' ')
        exit(1)

    # load Embedding matrix
    row_nb = index_from+len(vocab_idmap)    
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (row_nb, vec_size) )

    wrd_found = {}

    for line in f: # read from the emb file
        all_ent   = line.split()
        word, vec = all_ent[0].lower(), map (float, all_ent[1:])

        if vocab_idmap.has_key(word):
            wrd_found[word] = 1 
            wid    = vocab_idmap[word] + index_from
            E[wid] = np.array(vec)

    f.close()
    print " Number of words found in emb matrix: " + str (len (wrd_found)) + " of " + str (len(vocab_idmap))

    return E        


def get_embeddings_from_weight_file(weight_file_path, vocab_idmap, out_vocab_file, index_from=3):

    """
    Extract word embeddings from the model file
    Args:
      weight_file_path (str) : Path to the file to analyze
    """

    print "Loading pre-trained embeddings from out-domain model ......"

    import h5py

    #read out-model vocab
    out_vocab_idmap = cPickle.load(open(out_vocab_file, "rb"))    

    # get model embeddings
    f = h5py.File(weight_file_path)
    try:
        if len(f.items())==0:
            print "Invalid model file.."
            exit(1)
        E_out   = f.items()[0][1]['param_0'][:]

    finally:
        f.close()

    print " Shape of out-domain emb matrix: " + str (E_out.shape)

    # init Embedding matrix
    row_nb   = index_from+len(vocab_idmap)    
    vec_size = E_out.shape[1]
    E        = 0.01 * np.random.uniform( -1.0, 1.0, (row_nb, vec_size) )

    # map embeddings..
    E[0:index_from] = E_out[0:index_from]

    wrd_found = {}
    for awrd in vocab_idmap.keys():
        wid = vocab_idmap[awrd] + index_from # wid in the in-dom vocab
        if out_vocab_idmap.has_key(awrd):
            wrd_found[awrd] = 1 
            wid_out = out_vocab_idmap[awrd] + index_from # wid in the out-dom emb
            E[wid]  = E_out[wid_out]

    print " Number of words found in emb matrix: " + str (len (wrd_found)) + " of " + str (len(vocab_idmap))
    print " Shape of emb matrix: " + str (E.shape)

    return E    


def remap_labels(y, merge_labels=None):

    if not merge_labels:
        return y

    y_modified = []
    for alabel in y:
        if merge_labels.has_key(alabel):
            y_modified.append(merge_labels[alabel])
        else:
            y_modified.append(alabel)

    return y_modified        


def adjust_index(X, labels, start_char=1, index_from=3, maxlen=None):

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if start_char is not None: # add start of sentence char
        X = [[start_char] + [w + index_from for w in x] for x in X] # shift the ids to index_from; id 3 will be shifted to 3+index_from
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen: # exclude tweets that are larger than maxlen
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)

        X      = new_X
        labels = new_labels

    return (X, labels)     


def numberize_sentences(sentences, vocab_idmap, oov_char=2):  

    sentences_id=[]  

    for sid, sent in enumerate (sentences):
        tmp_list = []
        for wrd in sent.split():
            wrd_id = vocab_idmap[wrd] if vocab_idmap.has_key(wrd) else oov_char 
            tmp_list.append(wrd_id)

        sentences_id.append(tmp_list)

    return sentences_id    


def numberize_labels(all_str_label, label_id_map):

    label_cnt = {}
    labels    = []
    
    for a_label in all_str_label:
        labels.append(label_id_map[a_label])

        if label_cnt.has_key(a_label):
            label_cnt[a_label] += 1
        else:
            label_cnt[a_label] = 1    

    return (labels, label_cnt)        



def get_label(str_label):
    if  str_label == "informative":
        return 1
    elif str_label == "not informative":
        return 0
    else:
        print "Error!!! unknown label " + str_label
        exit(1)        


