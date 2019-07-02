# -*- coding: utf-8 -*-
import io
from collections import Counter
import itertools as its

from gensim.models.fasttext import FastText





class Editor():
    def __init__(self, mod_txt_spm, mod_txt_jieba,cls_txt_spm,cls_txt_jieba, min_count=5):
        self.mod_spm = self.load(mod_txt_spm, min_count, 'mod_spm')
        self.mod_jieba = self.load(mod_txt_jieba, min_count, 'mod_jieba')

        #todo: classic chinese
        self.cls_spm = self.load(cls_txt_spm, min_count, 'cls_spm')
        self.cls_jieba = self.load(cls_txt_jieba, min_count, 'cls_jieba')

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

        split_at_1 = round(len(sents) * 0.8)
        training = sents[:split_at_1]
        test = sents[split_at_1:]

        model['w2i'] = w2i
        model['i2w'] = i2w
        model['vocab_size'] = len(w2i)
        model['training'] = training
        model['test'] = test

    #    model['training'] = [w2i.get(word, len(w2i)+1) for sent in training for word in sent]
    #    model['dev'] = [w2i.get(word, len(w2i)+1) for sent in dev for word in sent]
    #    model['test'] = [w2i.get(word, len(w2i)+1) for sent in test for word in sent]

        return model

    @classmethod
    def clean_lines(cls, textfile, model_type):
        if model_type in ['mod_jieba', 'mod_spm', 'cls_jieba', 'cls_spm']:
            sentences = []
            with open(textfile, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    line = (line.lstrip('▁').strip()) if (model_type == 'mod_spm' or model_type == 'cls_spm') else line
                    sentence = line.split(' ')
                    sentence.insert(0, '▁')
                    sentences.append(sentence)
            return sentences
        else:
            raise ValueError('invalid model type')


def daten_teilen(segmented_text):  # TODO derzeit unbenutzt
    """ takes a long text and splits it line-wise into three parts """
    lines = segmented_text.split('\n')
    split_at_1 = round(len(lines) * 0.8)
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
    editor = Editor('../data/result/segmented_01_spm.txt', '../data/result/segmented_01_jieba.txt','../data/result/segmented_02_spm.txt', '../data/result/segmented_02_jieba.txt', 5)
    #mod_spm = editor.mod_spm['vocab_size']
    #mod_jieba = editor.mod_jieba['vocab_size']
    #cls_spm = editor.cls_spm['vocab_size']
    #cls_jieba = editor.cls_jieba['vocab_size']
#    print(editor.mod_jieba['test'])
    print(editor.mod_spm['test'])
    



 #   print('\nSPM sentences for Modern Chinese:', mod_spm, '\nJieba Editor for Modern Chinese:', mod_jieba)
 #   print('\nSPM sentences for Classic Chinese:', cls_spm, '\nJieba Editor for Classic Chinese:', cls_jieba)

#todo: 1.data seperated to traing, dev, and tes
#todo: 2.evaluation
#todo: 3 word embbedding input and as gold , how
#todo: last:visulation
