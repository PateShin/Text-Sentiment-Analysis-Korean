# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14ohm3LwItuaFtJw_MPrQvjuUBGtHG2II
"""

from tensorflow.keras.utils import get_file

train_fname = 'ratings_train.tsv'
test_fname = 'ratings_test.tsv'
train_origin = 'https://raw.github.com/e9t/nsmc/master/ratings_train.txt'
test_origin = 'https://raw.github.com/e9t/nsmc/master/ratings_test.txt'

train_path = get_file(train_fname, train_origin)
test_path = get_file(test_fname, test_origin)

import pandas as pd
import numpy as np

train_df = pd.read_csv(train_path, sep='\t') # tsv file
train_df.head()

test_df = pd.read_csv(test_path, sep='\t') # tsv file
test_df.head()

train_df.isnull().any()

train_df = train_df.dropna(axis=0).reset_index(drop=True)

test_df.isnull().any()

test_df = test_df.dropna(axis=0).reset_index(drop=True)

print('Train data shape: ', train_df.shape)
n_lebel = len(train_df[train_df.label == 0])
print('Label 0 in Train data: {} ({:.1f}%)'.format(n_lebel, n_lebel*100/len(train_df)))
n_lebel = len(train_df[train_df.label == 1])
print('Label 1 in Train data: {} ({:.1f}%)'.format(n_lebel, n_lebel*100/len(train_df)))

print('\nTest data shape: ', test_df.shape)
n_lebel = len(test_df[test_df.label == 0])
print('Label 0 in Test data: {} ({:.1f}%)'.format(n_lebel, n_lebel*100/len(test_df)))
n_lebel = len(test_df[test_df.label == 1])
print('Label 1 in Test data: {} ({:.1f}%)'.format(n_lebel, n_lebel*100/len(test_df)))

train_df = train_df[['document', 'label']]
test_df = test_df[['document', 'label']]

!java -version

# Commented out IPython magic to ensure Python compatibility.
# %pip install PyKomoran

"""# Testing"""

from PyKomoran import *

corpus = "① 대한민국은 민주공화국이다."
komoran = Komoran("STABLE")
komoran.get_plain_text(corpus)

"""# Tokenize the texts
## Train Data frame
"""

# As there was problem in 68388 I have narrowed that data points
# In real situation it will be `len(train_df['documentation])` instead of
# `(68389, 68400)`
for i in range(68380, 68400):
  try:
    komoran.get_list(train_df['document'][i])
  except:
    train_df['document'][i] = None

train_df[68381: 68390]

train_df.isnull().any()

train_df = train_df.dropna(axis=0).reset_index(drop=True)

len(train_df['document'])

stop_pos_tags =  ['IC', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX',
                   'EF', 'ETN', 'ETM', 'XSA', 'SF', 'SP', 'SS', 'SE', 'SO', 'SL', 'SH',
                   'SW', 'NF', 'NV', 'SN', 'NA']

def tokenize(corpus, stop_pos_tags):
    result = []
    pairs = komoran.get_list(corpus)
    for pair in pairs:
        morph = pair.get_first()
        pos = pair.get_second()
        if pos not in stop_pos_tags:
            if pos in ['VV', 'VA', 'VX', 'VCP', 'VCN']:
                morph = morph + '다'
            result.append(morph)
    return result

tokens_list = []

for i in range(len(train_df['document'])):
    tokens_list.append(tokenize(train_df['document'][i], stop_pos_tags))

train_df['tokens'] = tokens_list

train_df.head()

train_df = train_df[train_df['tokens'].str.len() > 2]

len(train_df['document'])

"""## Test dataframe"""

# original
len(test_df['document'])

for i in range(len(test_df['document'])):
  try:
    komoran.get_list(test_df['document'][i])
  except:
    test_df['document'][i] = None

len(test_df['document'])

test_df.isnull().any()

test_df = test_df.dropna(axis=0).reset_index(drop=True)

tokens_list = []

for i in range(len(test_df['document'])):
    tokens_list.append(tokenize(test_df['document'][i], stop_pos_tags))

test_df['tokens'] = tokens_list

test_df.head()

test_df = test_df[test_df['tokens'].str.len() > 2]

"""# LTSM"""

from tensorflow.keras.preprocessing.text import Tokenizer
import os
import pickle

tokenizer_name = 'keras_naver_review_tokenizer.pickle'
save_path = os.path.join(os.getcwd(), tokenizer_name)

max_words = 35000
tokenizer = Tokenizer(num_words=max_words, oov_token = True)
tokenizer.fit_on_texts(train_df.tokens)
train_df.tokens = tokenizer.texts_to_sequences(train_df.tokens)
test_df.tokens = tokenizer.texts_to_sequences(test_df.tokens)

with open(save_path, 'wb') as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

train_df.head()

test_df.head()

X_train = train_df.tokens
Y_train = train_df.label

X_test = test_df.tokens
Y_test = test_df.label

print('X_train shape: ', X_train.shape)
print('Y_train shape: ', Y_train.shape)
print('\nX_test shape: ', X_test.shape)
print('Y_test shape: ', Y_test.shape)

from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len=40
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
# Train
batch_size = Y_train.shape[0]
input_dim = 1
Y_train = encoder.fit_transform(Y_train) # Labeling
Y_train = np.reshape(Y_train, (batch_size, input_dim)) # Reshape
# Test
batch_size = Y_test.shape[0]
Y_test = encoder.transform(Y_test) # Labeling
Y_test = np.reshape(Y_test, (batch_size, input_dim)) # Reshape

print(Y_train.shape)
print(Y_test.shape)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(Embedding(max_words, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

"""# Training"""

hist = model.fit(X_train, Y_train, batch_size=32, epochs=5)

"""## Model Testing"""

loss, acc = model.evaluate(X_test, Y_test, batch_size=32)

print('Test loss:', loss)
print('Test accuracy:', acc)

import os

save_dir = os.getcwd()
model_name = 'keras_naver_review_trained_model.h5'

# Save model and weights
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

"""# Final Solution"""

from  tensorflow.keras.models import load_model
import os
import pickle

def load_tokenizer(path):
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

load_dir = os.getcwd()
model_name = 'keras_naver_review_trained_model.h5'
tokenizer_name = 'keras_naver_review_tokenizer.pickle'
model_path = os.path.join(load_dir, model_name)
tokenizer_path = os.path.join(load_dir, tokenizer_name)

model = load_model(model_path)
tokenizer = load_tokenizer(tokenizer_path)

import numpy as np
from PyKomoran import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len=40
komoran = Komoran("STABLE")
stop_pos_tags =  ['IC', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX',
                   'EF', 'ETN', 'ETM', 'XSA', 'SF', 'SP', 'SS', 'SE', 'SO', 'SL', 'SH',
                   'SW', 'NF', 'NV', 'SN', 'NA']

def tokenize(corpus, stop_pos_tags):
    result = []
    pairs = komoran.get_list(corpus)
    for pair in pairs:
        morph = pair.get_first()
        pos = pair.get_second()
        if pos not in stop_pos_tags:
            if pos in ['VV', 'VA', 'VX', 'VCP', 'VCN']:
                morph = morph + '다'
            result.append(morph)
    return result

def predict_sentiment(text, model):
    tokens = []
    tokens.append(tokenize(text, stop_pos_tags))
    tokens = tokenizer.texts_to_sequences(tokens)
    x_test = pad_sequences(tokens, maxlen=max_len)
    predict = model.predict(x_test)
    if predict[0] > 0.5:
        return 'GOOD'
    else:
        return 'BAD'

review_text = '재밌당.'
result = predict_sentiment(review_text, model)

print('{} : {}'.format(review_text, result))