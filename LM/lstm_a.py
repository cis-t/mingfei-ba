# -*- coding: utf-8 -*-
import io
import itertools  as its
import tensorflow as tf
import numpy as np
from collections import Counter
from keras.models import Sequential,load_model
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models.fasttext import FastText


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


def clean_lines(txt_):
    sents = []
    with open(txt_, "r") as f:
        for l in f:
            if l[0] == "#": continue
            l = l.strip().split("\t")[-1].split(" ")
            if len(l) == 0: continue
            sents.append(l)
    print("Obtained {} verses from {}.".format(len(sents), txt_))
    return sents

#todo: py2 __init__
def editor(txt_, min_count=5):
    sents = clean_lines(txt_)
    w2c = Counter(" ".join(
        its.chain.from_iterable(sents)).split(" ")).most_common()
    w2i = {}
    for idx, (w, c) in enumerate(w2c):
        if c < min_count: break
        w2i[w] = idx
    w2i["<unk>"] = len(w2i)
    w2i["</s>"] = len(w2i)
    i2w = {v: k for k, v in w2i.items()}
    w2i, i2w = w2i, i2w
    vocab_size = len(w2i)



#todo: von Mengjie
def get_lm_model(vocab_size, flat_len, emb_dim, lstm_dim):
    model = Sequential()
    model.add(Embedding(vocab_size, emb_dim, input_length=flat_len))
    model.add(LSTM(lstm_dim))
    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    print(model.summary())
    return model


if __name__ == "__main__":
    MAX_LEN = 50
    FLAT_LEN = 5
    MIN_COUNT = 5
    EMB_DIM = 100
    LSTM_DIM = 100
    #todo: daten übergeben??
    edition = BibleEdition('../data/result/segmented_01_jieba.txt', min_count=MIN_COUNT)
    dl = DataLoader(edition, max_len=MAX_LEN)

    
    lm_model = get_lm_model(
        vocab_size=edition.vocab_size,
        flat_len=FLAT_LEN,
        emb_dim=EMB_DIM,
        lstm_dim=LSTM_DIM)

    NUM_EPOCH = 100
    BATCH_SIZE = 1024
    VAL_SPLIT = 0.2
    SAVED_PATH = "saved_model.pkl"

    inference = True

    if not inference:
        stop = EarlyStopping(  # vermeidung von Overfitting
            monitor="val_loss", min_delta=0,
            patience=5, verbose=1, mode="auto")
        saver = ModelCheckpoint(
            SAVED_PATH, monitor="val_loss",
            verbose=0, save_best_only=True)
        history = lm_model.fit(
            dl.X, dl.y,
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCH,
            verbose=1,
            validation_split=VAL_SPLIT,
            callbacks=[stop, saver])
    else:
        lm_model = load_model(SAVED_PATH)
        gold_tokens, pred_tokens = [], []
        for step in range(1000):
            inp = dl.X[step]
            pred = np.argmax(lm_model.predict(np.asarray([inp])))
            gold_tokens.append(edition.i2w[dl._y[step]])
            pred_tokens.append(edition.i2w[pred])
            if step % 50 == 0:
                print("gold:")
                print(" ".join(gold_tokens))
                print("pred:")
                print(" ".join(pred_tokens))
                print("=" * 20)
                gold_tokens, pred_tokens = [], []