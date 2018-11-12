import warnings
warnings.filterwarnings('ignore')

import numpy as np
from collections import defaultdict
from Bio import SeqIO
from nltk import trigrams
import gensim

import sys
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from keras.layers import Dropout
from keras.layers import Input, Dense, Lambda, LSTM, RepeatVector, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras import regularizers
from keras.layers import GaussianNoise
from keras.layers import Activation
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D, GRU
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.models import load_model


seed = 42
np.random.seed(seed)
MAX_SEQUENCE_LENGTH = 302
MAX_NB_WORDS = 10334
EMBEDDING_DIM = 200

new_model = gensim.models.Word2Vec.load('my_wordvec_model_trembl_size_200')

def create_model_no_pretrain_bidirec():
    print ("\n")
    print ("Building the Neural Network model")
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), )
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(GRU(32, dropout=0.5, recurrent_dropout=0.1, return_sequences = True))(embedded_sequences) 
    x = Bidirectional(GRU(32, dropout=0.5,recurrent_dropout=0.1))(x)
    preds = Dense(1, activation='sigmoid')(x)

    new_model = Model(sequence_input, preds)
    new_model.compile(loss='binary_crossentropy',
                  optimizer= 'adam',
                  metrics=['acc'])
    
    return new_model

texts = []
for index, record in enumerate(SeqIO.parse('primary_bacteriocin_training_set', 'fasta')):
    tri_tokens = trigrams(record.seq)
    temp_str = ""
    for item in ((tri_tokens)):
        #print(item),
        temp_str = temp_str + " " +item[0] + item[1] + item[2]
    texts.append(temp_str)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS) #MAX_NB_WORDS = 10334
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index


# Including all trigrams in word_index
count = len(word_index)
for index, item in enumerate(new_model.wv.vocab):
    if item.lower() not in word_index:
        count = count + 1
        word_index[item.lower()] = count 


# prepare embedding matrix
num_words = MAX_NB_WORDS 
#in our wild data set never seen trigrams may come, so we have to include them
embedding_matrix = np.zeros((num_words+1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    #print (word)
    if word.upper() in new_model.wv.vocab:
        embedding_vector = new_model[word.upper()]
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

final_model = None
final_model = create_model_no_pretrain_bidirec()
final_model.load_weights('final_NN_model.h5')


if __name__ == '__main__':
    #print ('hello world')
    target_fasta = sys.argv[1] 

    hansky_texts = []
    keep_track_index = []
    for index, record in enumerate(SeqIO.parse(target_fasta, 'fasta')):
        if len(record.seq) <= 302:
            keep_track_index.append(index)
            tri_tokens = trigrams(record.seq)
            temp_str = ""
            for item in ((tri_tokens)):
                #print(item),
                temp_str = temp_str + " " +item[0] + item[1] + item[2]
            #print (temp_str)
            hansky_texts.append(temp_str)

    
    
    hansky_sequences = tokenizer.texts_to_sequences(hansky_texts)
    hansky_data = pad_sequences(hansky_sequences, maxlen=MAX_SEQUENCE_LENGTH) 


    hansky_scores_probability = final_model.predict(hansky_data)
    hansky_scores = np.where(hansky_scores_probability > 0.5, 1, 0)
    print ("There are total %s sequences" % str(len(hansky_scores_probability)))
    #print (len(hansky_scores_probability))
    #print (len(hansky_scores[hansky_scores == 1]))
    print ("\n")
    print ("Predicting bacteriocins")
    print ("There are %s bacteriocins with probability of >= 0.95" %
                 str(len(hansky_scores_probability[hansky_scores_probability >= 0.95])))
    #print (len(hansky_scores_probability[hansky_scores_probability >= 0.9]))
    #print (hansky_scores_probability)

    out_handle = open(target_fasta+'_results', 'w')

    print ("Creating result file")
    count = 0
    for index, record in enumerate(SeqIO.parse(target_fasta, 'fasta')):
        if index in keep_track_index:
            out_handle.write(">%s|%s\n%s\n" % (record.description, 
                            str(hansky_scores_probability[count][0]), 
                            record.seq))
            count = count + 1

    out_handle.close()
    print ("Done!! ")

    
