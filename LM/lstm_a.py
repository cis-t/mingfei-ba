# -*- coding: utf-8 -*-

from collections import Counter
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D
import io
import itertools
from gensim.models.fasttext import FastText
import tensorflow as tf

def fileread(filename):
    """
    segmented files einlesen
    get list of lists
    """
    with open(filename, 'r') as f:
        sentences = []
        current_sentence = []
        for line in f:
            if line == '\n':
                sentences.append(current_sentence)
                current_sentence = []
            else:
                current_sentence.append(line.strip())
        sentences.append(current_sentence)
    return sentences

def daten_teilen(segmented_text):  # TODO derzeit unbenutzt
    """ takes a long text and splits it line-wise into three parts """
    lines = segmented_text.split('\n')
    split_at_1 = round(len(lines) * 0.6)
    split_at_2 = round(len(lines) * 0.8)
    training = lines[:split_at_1]
    dev = lines[split_at_1:split_at_2]
    test = lines[split_at_2:]
    return (training, dev, test)

def fastText_embedding(daten):
    """
    get embedding for training, dev, test daten
    :param daten:
    :return:
    """
    ll_daten = fileread(daten)
    model = FastText(size=4, window=3, min_count=1)
    model.build_vocab(sentences=ll_daten)
    model.train(sentences=ll_daten, total_examples=len(ll_daten), epochs=10)  # train
    model.wv.save_word2vec_format('daten_embedding.model')
    return model

def load_fastText_word(fname):
    """
    get pretrained fastText vocabulary,
    get also possible data = {vocal,vecs})
    https://fasttext.cc/docs/en/crawl-vectors.html
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    """
     #with this code line
    #{'2000000': <map object at 0x114f9e4a8>, 
    # '，': <map object at 0x114fa17b8>, 
    # '的': <map object at 0x114fa6ac8>,  ...}

    #without this code line
    #{'，': <map object at 0x114fa46d8>, 
    # '的': <map object at 0x1153d19e8>, 
    # '。': <map object at 0x1153d3cf8>, 
    # '</s>': <map object at 0x1153d8080>, 
    # '、': <map object at 0x1153dc390>,  ...}
    """
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    vocal = [w for w,vec in data.items()]
    return vocal


# Hyperparameter
VOCAB_SIZE = 10000
MAX_LEN = 100
BATCH_SIZE = 50
EPOCHS = 3
LOSS_FUNCTION = 'binary_crossentropy'
OPTIMIZER = 'adam'
UNKNOWN_TOKEN = '<unk>'



def create_dictionary(texts, vocab_size):
    """
    Creates a dictionary that maps words to ids. More frequent words have lower ids.
    The dictionary contains at the vocab_size-1 most frequent words (and a placeholder '<unk>' for unknown words).
    The place holder has the id 0.
    
    Input is a list of texts, each text being represented as a list of tokens
    """
    wordcounter = Counter()
    for text in texts:
        wordcounter.update(text)
    vocab = [w for w, _ in wordcounter.most_common(vocab_size - 1)]
    mapping = {w : c for (c, w) in enumerate(vocab, start=1)}
    mapping[UNKNOWN_TOKEN] = 0
    return mapping

def to_ids(words, dictionary):  # TODO derzeit unbenutzt
    """ Takes a list of words and converts them to ids using the word2id dictionary. """
    return [dictionary.get(w, 0) for w in words]


def build_and_evaluate_model(iddict, train_x, train_y, dev_x, dev_y, test_x, test_y):
    """
    Builds, trains and evaluates a keras LSTM model.
    Returns the score of the loss function and accuracy on the test data, as well as the trained model itself
    """
    train_x = sequence.pad_sequences([to_ids(doc, iddict) for doc in train_x], maxlen=MAX_LEN)
    dev_x = sequence.pad_sequences([to_ids(doc, iddict) for doc in dev_x], maxlen=MAX_LEN)
    test_x = sequence.pad_sequences([to_ids(doc, iddict) for doc in test_x], maxlen=MAX_LEN)
    standard_vectors = load_vectors('../fastText/[ch]aa')

    # initialize model
    model = Sequential()
    # Add layers
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=80))
    # TODO randomly initalize, update during training of LSTM, perhaps look here - https://bit.ly/2MKKTfl
    # TODO requires loading the pretrained vectors into numpy matrix first
    model.add(Bidirectional(LSTM(units=25)))
    model.add(Dense(1, activation='sigmoid')) # TODO compare to standard_vectors
    # Compile model
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=['accuracy'])
    # Fit to (train on) training data while validating on development data
    model.fit(x=train_x, y=train_y, validation_data=(dev_x, dev_y), batch_size=BATCH_SIZE, epochs=EPOCHS)
    # Evaluate model
    loss_score, accuracy_score = model.evaluate(test_x, test_y)
    
    return loss_score, accuracy_score, model


def main():
    print('Loading data...')
#    train_x, train_y, dev_x, dev_y, test_x, test_y = 0  # TODO (x) must each be a list of texts, each text being represented as a list of tokens / (y) must each be a list of labels

    train_x = fileread('../data/result/segmented_01_jieba.txt')
    train_y = [0 for _ in train_x]
    dev_x = fileread('../data/result/segmented_01_spm.txt')
    dev_y = [0 for _ in dev_x]
    test_x = dev_x
    test_y = dev_y

    word_ids = create_dictionary((train_x + dev_x + test_x), VOCAB_SIZE)
    print(len(train_x), 'training samples')
    print(len(dev_x), 'development samples')
    
    loss_score, accuracy_score, model = build_and_evaluate_model(word_ids, train_x, train_y, dev_x, dev_y, test_x, test_y)
    print('')
    print(LOSS_FUNCTION, 'score on test data:', loss_score)
    print('accuracy on test data:', accuracy_score)


if __name__ == '__main__':
    main()
