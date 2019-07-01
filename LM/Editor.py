# -*- coding: utf-8 -*-
import io
from collections import Counter
import itertools as its

from gensim.models.fasttext import FastText



class Editor():
    def __init__(self, txt_spm, txt_jieba, min_count=5):
        self.spm = self.load(txt_spm, min_count, 'spm')
        self.jieba = self.load(txt_jieba, min_count, 'jieba')

    def load(self, txt_, min_count, model_type):
        model = {}
        sents = self.clean_lines(txt_, model_type)


        w2c = Counter(" ".join(its.chain.from_iterable(sents)).split(" ")).most_common()
        w2i = {}
        for idx, (w, c) in enumerate(w2c):
            if c < min_count:
                break
            w2i[w] = idx
        w2i["<unk>"] = len(w2i)
        w2i["</s>"] = len(w2i)
        i2w = {v: k for k, v in w2i.items()}


        #todo: seperate daten???
        model['sents'] = sents
        split_at_1 = round(len(sents) * 0.6)
        split_at_2 = round(len(sents) * 0.8)
        training = sents[:split_at_1]
        dev = sents[split_at_1:split_at_2]
        test = sents[split_at_2:]

        model['dev'] = dev
        model['training'] = training
        model['test'] = test
        model['w2i'] = w2i
        model['i2w'] = i2w
        model['vocab_size'] = len(w2i)

        return model

    @classmethod
    def clean_lines(cls, textfile, model_type):
        if model_type == 'jieba':
            sentences = []
            with open(textfile, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    sentence = line.split(' ')
                    sentences.append(sentence)
            return sentences
        elif model_type == 'spm':
            sentences = []
            with open(textfile, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    line = line.lstrip('▁').strip()
                    sentence = line.split(' ')
                    sentences.append(sentence)
            return sentences
        else:
            raise ValueError('invalid model type')


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



if __name__ == '__main__':
    editor = Editor('../data/result/segmented_01_spm.txt', '../data/result/segmented_01_jieba.txt', 5)
    spm = editor.spm['vocab_size']
    jieba = editor.jieba['vocab_size']
    print('SPM sentences:', spm, '\nJieba sentences:', jieba)

#todo: 1.data seperated to traing, dev, and tes
#todo: 2.evaluation
#todo: 3 word embbedding input and as gold , how
#todo: last:visulation
