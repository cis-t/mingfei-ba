import numpy as np
from keras.utils import to_categorical
import LM.Editor as editor

class DataLoader():
    def __init__(self, editor, model_type, flat_len=5):
        if model_type == 'mod_spm':
            editor = editor.mod_spm
        elif model_type == 'mod_jieba':
            editor = editor.mod_jieba
        elif model_type == 'cls_spm':
            editor = editor.mod_jieba
        elif model_type == 'cls_jieba':
            editor = editor.mod_jieba
        else:
            raise ValueError('invalid model type')

        #todo: training data -> one-hot vectorization
        self.train_X, self.train_Y, self.train_original_Y = self.get(editor, 'training', flat_len)
#        self.dev_X, self.dev_Y, self.dev_original_Y = self.get(editor, 'dev', flat_len)
        self.test_X, self.test_Y, self.test_original_Y = self.get(editor, 'test', flat_len)


    @classmethod
    def get(cls, editor, type, flat_len):
        X, Y = [], []
        for sent in editor[type]:
            if len(sent) == 0:
                continue
            sent_ids = []
            for w in sent:
                try:
                    sent_ids.append(editor['w2i'][w])
                except KeyError:
                    sent_ids.append(editor['w2i']["<unk>"])
            for i in range(0, len(sent_ids) - flat_len):
                X.append(sent_ids[i:i + flat_len])
                Y.append(sent_ids[i + flat_len])
        return np.asarray(X), to_categorical(Y, editor['vocab_size']), Y

"""
    @classmethod
    def get(cls, editor, type, flat_len, max_len):
        X, Y = [], []
        for sent in editor[type]:
            if len(sent) < max_len:
                sent += ["</s>"] * (max_len - len(sent))
            else:
                sent = sent[:max_len - 1] + ["</s>"]
            sent_ids = []
            for w in sent:
                try:
                    sent_ids.append(editor['w2i'][w])
                except KeyError:
                    sent_ids.append(editor['w2i']["<unk>"])
            for i in range(0, len(sent_ids) - flat_len, flat_len):
                X.append(sent_ids[i:i + flat_len])
                Y.append(sent_ids[i + flat_len])
        return np.asarray(X), to_categorical(Y, editor['vocab_size']), Y
"""

if __name__ == '__main__':
    editor = editor.Editor('../data/result/segmented_01_spm.txt', '../data/result/segmented_01_jieba.txt','../data/result/segmented_02_spm.txt', '../data/result/segmented_02_jieba.txt', 5)
    dl_mod_spm = DataLoader(editor, 'mod_spm', 5)
    dl_mod_jieba = DataLoader(editor,'mod_jieba', 5)
    dl_cls_spm = DataLoader(editor, 'cls_spm', 5)
    dl_cls_jieba = DataLoader(editor,'cls_jieba', 5)

#    print('\n\n Modern SPM')
#    print(dl_mod_spm.train_X, dl_mod_spm.train_Y, dl_mod_spm.train_original_Y)
#    print('\n\n Modern JIEBA')
#    print(dl_mod_jieba.train_X, dl_mod_jieba.train_Y, dl_mod_jieba.train_original_Y)
#    print('\n\n CLASSIC SPM')
#    print(dl_cls_spm.train_X, dl_cls_spm.train_Y, dl_cls_spm.train_original_Y)
#    print('\n\n CLASSIC JIEBA')
#    print(dl_cls_jieba.train_X, dl_cls_jieba.train_Y, dl_cls_jieba.train_original_Y)




