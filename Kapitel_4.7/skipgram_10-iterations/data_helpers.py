import numpy as np
import re
import itertools
from collections import Counter
import pickle

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files


#-------------------- Generate TRAIN subst --------------------------------------
    positive_examples_lowerhalf = list(open("./data/positive_lowerhalf_shuffled_exchange.txt").readlines())
    positive_examples_lowerhalf = [s.strip() for s in positive_examples_lowerhalf]

    negative_examples_lowerhalf = list(open("./data/negative_lowerhalf_shuffled_exchange.txt").readlines())
    negative_examples_lowerhalf = [s.strip() for s in negative_examples_lowerhalf]

    # Split by words
    x_text_subst = positive_examples_lowerhalf + negative_examples_lowerhalf
    x_text_subst = [clean_str(sent) for sent in x_text_subst]
    x_text_subst = [s.split(" ") for s in x_text_subst]

    
    positive_labels_subst = [[0, 1] for _ in positive_examples_lowerhalf]
    negative_labels_subst = [[1, 0] for _ in negative_examples_lowerhalf]
    y_subst = np.concatenate([positive_labels_subst, negative_labels_subst], 0)


#-------------------- Generate TEST  --------------------------------------
    positive_examples_upperHalf = list(open("./data/positive_upperhalf_shuffled_800.txt").readlines())
    positive_examples_upperHalf = [s.strip() for s in positive_examples_upperHalf]

    negative_examples_upperHalf = list(open("./data/negative_upperhalf_shuffled_800.txt").readlines())
    negative_examples_upperHalf = [s.strip() for s in negative_examples_upperHalf]

    # Split by words
    x_text_upper = positive_examples_upperHalf + negative_examples_upperHalf
    x_text_upper = [clean_str(sent) for sent in x_text_upper]
    x_text_upper = [s.split(" ") for s in x_text_upper]

    # Generate labels TEST subst
    positive_labels_upper = [[0, 1] for _ in positive_examples_upperHalf]
    negative_labels_upper = [[1, 0] for _ in negative_examples_upperHalf]
    y_upper = np.concatenate([positive_labels_upper, negative_labels_upper], 0)

    return [x_text_subst, y_subst, x_text_upper, y_upper]


def pad_sentences(sentences, max_length, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """

    sequence_length = max_length
    print("Print sequence_length: ", sequence_length)

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))

    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]

    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """

    sentences_train, labels_train, sentences_test, labels_test = load_data_and_labels()

    sentences_all = np.concatenate([sentences_train, sentences_test], 0)

    max_length = max(len(x) for x in sentences_all)
    sentences_all_padded = pad_sentences(sentences_all, max_length)

    sentences_train_padded = pad_sentences(sentences_train, max_length)
    sentences_test_padded = pad_sentences(sentences_test, max_length)

    vocabulary, vocabulary_inv = build_vocab(sentences_all_padded)

    x_train, y_train = build_input_data(sentences_train_padded, labels_train, vocabulary)
    x_test, y_test = build_input_data(sentences_test_padded, labels_test, vocabulary)

    return [x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
