import os
import pickle

import numpy as np
import data_helpers
from w2v import train_word2vec

import tensorflow as tf

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence

from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.contrib.tensorboard.plugins import projector

import matplotlib.pyplot as plt

np.random.seed(0)

# ---------------------- Parameters section -------------------

# Data source
data_source = "local_dir"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64

num_epochs = 500

# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

#
# ---------------------- Parameters end -----------------------

def load_data(data_source, datalength):
    x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    y = y.argmax(axis=1)

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]

    x = x[:datalength]
    y = y[:datalength]

    # use 80% of data for training
    train_len = int(len(x) * 0.8)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary_inv



train_scores = []
test_scores  = []

for datalength in [100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:

    train_scores_run = []
    test_scores_run  = []

    for run in range(0, 10, 1):
        # Data Preparation
        print("Load data...")
        x_train, y_train, x_test, y_test, vocabulary_inv = load_data(data_source, datalength)

        if sequence_length != x_test.shape[1]:
            print("Adjusting sequence length for actual size")
            sequence_length = x_test.shape[1]

        #--------------------------define CNN-non-static-----------------------------------------

        # train_word2vec returns embedding_weights of the trained word2vec embedding (see w2v.py)
        embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv,        num_features=embedding_dim, min_word_count=min_word_count, context=context)

        input_shape = (sequence_length,)
        model_input = Input(shape=input_shape)

        z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

        z = Dropout(dropout_prob[0])(z)

        # Convolutional block
        conv_blocks = []
        for sz in filter_sizes:
            conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        z = Dropout(dropout_prob[1])(z)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(1, activation="sigmoid")(z)

        model = Model(model_input, model_output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        #print(model.summary())

        # Initialize weights with word2vec
        weights = np.array([v for v in embedding_weights.values()])
        print("Initializing embedding layer with word2vec weights, shape", weights.shape)
        embedding_layer = model.get_layer("embedding")
        embedding_layer.set_weights([weights])

        #-------------------------------------------------------------------------------------

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=2)

        epoch_nr = 0
        val_loss = 1.00

        for epoch in range(0, num_epochs, 1):
            if (history.history['val_loss'][epoch] < val_loss):
                val_loss = history.history['val_loss'][epoch]
                epoch_nr = epoch
            else:
                pass

        train_scores_run.append(history.history['acc'][epoch_nr])
        test_scores_run.append(history.history['val_acc'][epoch_nr])

    train_scores.append(train_scores_run)
    test_scores.append(test_scores_run)

#--------------------------------------------------------------------------------------------------
train_sizes = [100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

print("val_loss_min: ", val_loss)
print("epoch_nr: ", epoch_nr)
print(train_scores)
print(test_scores)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test score")

plt.legend(loc="best")
plt.savefig('results.png')
plt.show()


#--------------------------------------------------------------------------------------------------
