# -*- coding: utf-8 -*-
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from collections import Counter
from gensim.models.fasttext import FastText
import tensorflow as tf
import io
import sys

# Hyperparameter
VOCAB_SIZE = 10000
MAX_LEN = 100
BATCH_SIZE = 50
EPOCHS = 10
LOSS_FUNCTION = 'binary_crossentropy'
OPTIMIZER = 'adam'
UNKNOWN_TOKEN = '<unk>'



















UNKNOWN_TOKEN = "<unk>"





def build_and_evaluate_model(train_x, train_y, dev_x, dev_y, test_x, test_y):
    """
    Builds, trains and evaluates a keras LSTM model.
    Returns the score of the loss function and accuracy on the test data, as well as the trained model itself
    """
    train_x = sequence.pad_sequences('\n'.join(train_x), maxlen=MAX_LEN)
    dev_x = sequence.pad_sequences('\n'.join(dev_x), maxlen=MAX_LEN)
    test_x = sequence.pad_sequences('\n'.join(test_x), maxlen=MAX_LEN)
    # TODO why are all labels in each set the same? shouldn't they be read from the original data?
    train_y = [0 for _ in train_x]
    dev_y = [1 for _ in dev_x]
    test_y = [2 for _ in test_x]
    
    model = Sequential()
    # Add layers
    model.add(Embedding())
    # TODO randomly initalize, update during training of LSTM, perhaps look here - https://bit.ly/2MKKTfl
    # requires loading the pretrained vectors into numpy matrix first
    model.add(Bidirectional(LSTM(units=25)))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=['accuracy'])
    # Fit to (train on) training data while validating on development data
    model.fit(x=train_x, y=train_y, validation_data=(dev_x, dev_y), batch_size=BATCH_SIZE, epochs=EPOCHS)
    # Evaluate model
    loss_score, accuracy_score = model.evaluate(test_x, test_y)
    return loss_score, accuracy_score, model


<<<<<<< HEAD


def build_and_evaluate_model(x_train, x_dev):
    """ Builds, trains and evaluates a keras LSTM model. """
    x_train = sequence.pad_sequences("\n".join(x_train), maxlen=MAX_LEN)
    x_dev = sequence.pad_sequences("\n".join(x_dev), maxlen=MAX_LEN)
    y_train = [0 for _ in x_train]
    y_dev = [1 for _ in x_dev]
    model = Sequential()

    # Add layers.
    #todo:model.add(Embedding())
    model.add(Bidirectional(LSTM(units=25)))
    # Compile model.
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # Fit to data.
    model.fit(x=x_train, y=y_train, validation_data=(x_dev, y_dev), batch_size=BATCH_SIZE, epochs=EPOCHS)

    score, acc = model.evaluate(x_dev, y_dev)
    return score, acc, model

def main(argv):
    print('Loading data...')
    x_train, y_train, x_dev, y_dev, word2id = ?????data(vocab_size=VOCAB_SIZE)
    print(len(x_train), 'training samples')
    print(len(x_dev), 'development samples')
    score, acc, _ = build_and_evaluate_model(x_train, y_train, x_dev, y_dev)
    print('\ndev score:', score)
    print('dev accuracy:', acc)


if __name__ == "__main__":

    main(sys.argv[1:])
=======
def main():
    print('Loading data...')
    train_x, train_y, dev_x, dev_y, test_x, test_y = 0  # TODO (x) must each be a list of texts, each text being represented as a list of tokens / (y) must each be a list of labels
    word_ids = create_dictionary((train_x + dev_x + test_x), VOCAB_SIZE)
    print(len(train_x), 'training samples')
    print(len(dev_x), 'development samples')
    
    loss_score, accuracy_score, model = build_and_evaluate_model(train_x, train_y, dev_x, dev_y, test_x, test_y)
    print('')
    print(LOSS_FUNCTION, 'score on training data:', loss_score)
    print('accuracy on training data:', accuracy_score)


if __name__ == '__main__':
    main()
>>>>>>> 0f7469072b23fb2b09752c30301b995d0575d65b
