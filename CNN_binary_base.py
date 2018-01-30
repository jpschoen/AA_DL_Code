# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-


from __future__ import print_function

import os
#import sys
import numpy as np
import pandas as pd
os.environ['KERAS_BACKEND']='tensorflow'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge
from keras.models import Model
#from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove_pretrained/glove.6B/'
TEXT_DATA_DIR = BASE_DIR 
MAX_SEQUENCE_LENGTH = 250
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding="utf-8")
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
ht_df = pd.read_csv("data_clean/ht_Sw_50.csv", encoding="utf-8")
print(list(ht_df))
ht_df = ht_df.loc[ht_df['post1_length'] >=50]
ht_df = ht_df.loc[ht_df['post2_length'] >=50]
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
labels = ht_df['matched']  # list of label ids
#for string in labels:
#    labels_index.setdefault(string, len(labels_index))
#labels =  np.asarray([labels_index[author] for author in labels])




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

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
# set up layers
convs = []
filter_sizes = [2,3,4,5]

for fsz in filter_sizes:
    l_conv = Conv1D(filters=128,kernel_size=fsz,activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)
    
x = Merge(mode='concat', concat_axis=1)(convs)
l_cov1= Conv1D(128, 5, activation='relu')(x)
l_pool1 = MaxPooling1D(5)(l_cov1)
drop = Dropout(0.1)(l_pool1)
l_cov2 = Conv1D(128, 5, activation='relu')(drop)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_flat = Flatten()(l_pool2)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)


model = Model(sequence_input, preds)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=5, batch_size=250)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model: CNN, 75%')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.savefig('CNN_50_acc.png', format='png', dpi=500)
