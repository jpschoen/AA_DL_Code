'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a Author Attribution model on Reddit data.

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

Reddit data can be found at:
https://www.reddit.com/r/datasets/comments/65o7py/updated_reddit_comment_dataset_as_torrents/

Borrowed code from https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
'''

from __future__ import print_function

import os
import io
#import sys
import numpy as np
import pandas as pd
os.environ['KERAS_BACKEND']='tensorflow'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, GlobalAveragePooling1D, concatenate
from keras.layers import Embedding
from keras.models import Model
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import itertools
#from theano.ifelse import ifelse
from sklearn.preprocessing import LabelEncoder
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.layers import Bidirectional

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove_pretrained/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/reddit_data/'
MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')


#read in  Data and list features
#read in  Data and list features
ht_df = pd.read_csv("data_clean/ht_subset_1.csv", encoding="utf-8")
print(list(ht_df))
print(ht_df.post1_length.describe())
#ht_df = ht_df.loc[ht_df['post1_length'] >=10]
#ht_df = ht_df.loc[ht_df['post2_length'] >=10]
#ht_df = ht_df.sample(n=850000, random_state=19)

posts1 = ht_df['ID1_post']  # list of text samples
posts2 = ht_df['ID2_post']  # list of text samples

# Vectorize posts
print('%s texts.' % len(posts1))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(posts1)
sequences = tokenizer.texts_to_sequences(posts1)

word_index = tokenizer.word_index
print('Found %s unique tokens in post 1' % len(word_index))

data1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(posts2)
sequences = tokenizer.texts_to_sequences(posts2)

word_index = tokenizer.word_index
print('Found %s unique tokens in post 2' % len(word_index))

data2 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Find difference matrix
data = abs(data1 - data2)



print('%s Unique Phone Numbers' % len(ht_df.ID1_phone.unique()))


labels_index = {}  # dictionary mapping label name to numeric id
labels = to_categorical(ht_df['matched'])  # list of label ids
#for string in labels:


#labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

#y_train = y_train.reshape((-1, 1))

#print('Rows: %d, columns: %d' % x_train.shape[0], x_train.shape[1] )

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)




print('Training model.')

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
drop = Dropout(0.5)(l_lstm)
preds = Dense(2, activation='softmax')(drop)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
model.summary()

callbacks = [EarlyStopping(monitor='val_acc', patience=2)]


history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
         epochs=80, verbose=1, batch_size=128, callbacks=callbacks)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Bidirectional LSTM model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


plt.savefig('binary_Bi_LSTM.png', format='png', dpi=500)

#plt.show()
