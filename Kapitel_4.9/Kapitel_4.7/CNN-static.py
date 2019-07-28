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
from tensorflow.python.keras.backend import clear_session


from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.contrib.tensorboard.plugins import projector

import matplotlib.pyplot as plt

# seed(0) makes the random numbers predictable
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

#--------------------------define CNN-static-----------------------------------------
def create_model(sequence_length):
    input_shape = (sequence_length, embedding_dim)

    model_input = Input(shape=input_shape)

    z = model_input

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

    print(model.summary())
    return model
#-------------------------------------------------------------------------------------

def load_data(data_source, datalength):

    x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv_list = data_helpers.load_data()

    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

    print("y_train: ", y_train[1])
    print("y_train argmax: ", y_train.argmax(axis=1))
    print("length of y_train argmax list: ", len(y_train.argmax(axis=1)))
    y_train = y_train.argmax(axis=1)

    print("y_test: ", y_test[1])
    print("y_test argmax: ", y_test.argmax(axis=1))
    print("length of y_test argmax list: ", len(y_test.argmax(axis=1)))
    y_test = y_test.argmax(axis=1)

    # Shuffle Train data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    print("x_test: ", len(x_test))
    print("y_test: ", len(y_test))

    # Shuffle Test data
    shuffle_indices = np.random.permutation(np.arange(len(y_test)))
    x_test = x_test[shuffle_indices]
    y_test = y_test[shuffle_indices]


    # trainlength = int(len(datalength) * 0.8)
    trainlength = int(datalength * 0.8)
    x_train = x_train[:trainlength]
    y_train = y_train[:trainlength]

    # testlength = int(len(datalength) * 0.2)
    testlength = int(datalength * 0.2)
    x_test = x_test[:testlength]
    y_test = y_test[:testlength]

    return x_train, y_train, x_test, y_test, vocabulary_inv



train_scores = []
test_scores  = []

for datalength in [100]:

    acc_nd = []
    loss_nd = []
    val_acc_nd = []
    val_loss_nd = []
    epoch_nr_nd = []

    for run in range(0, 10, 1):
        # Data Preparation
        print("Load data...")
        x_train, y_train, x_test, y_test, vocabulary_inv = load_data(data_source, datalength)

         # train_word2vec returns embedding_weights of the trained word2vec embedding (see w2v.py)
        embedding_weights = train_word2vec(datalength, np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim, min_word_count=min_word_count, context=context)

        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
        print("x_train static shape:", x_train.shape)
        print("x_test static shape:", x_test.shape)
        
        if sequence_length != x_test.shape[1]:
            print("Adjusting sequence length for actual size")
            sequence_length = x_test.shape[1]

        model = create_model(sequence_length)
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=2)

        epoch_nr = 0
        val_loss = 1.00

        for epoch in range(0, num_epochs, 1):
            if (history.history['val_loss'][epoch] < val_loss):
                val_loss = history.history['val_loss'][epoch]

                acc = history.history['acc'][epoch]
                loss = history.history['loss'][epoch]
                val_acc = history.history['val_acc'][epoch]
                epoch_nr = epoch
            else:
                pass

        acc_nd.append(acc)
        loss_nd.append(loss)
        val_acc_nd.append(val_acc)
        val_loss_nd.append(val_loss)
        epoch_nr_nd.append(epoch_nr)

        clear_session()
        del model

        with open("result_logs.txt", "a") as text_file:
            text_file.write("Samples: %s, Accuracy: %f, Loss: %f, Val.Acc: %f, Val.Loss: %f, Epoch: %d \n" % (datalength, acc, loss, val_acc, val_loss, epoch_nr))

    train_scores.append(acc_nd)
    test_scores.append(val_acc_nd)


print("acc_nd: ", acc_nd)
print("loss_nd: ", loss_nd)
print("val_acc_nd: ", val_acc_nd)
print("val_loss_nd: ", val_loss_nd)
print("epoch_nr_nd: ", epoch_nr_nd)


