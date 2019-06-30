import numpy as np
from keras.utils import to_categorical
import LM.Editor as editor

class DataLoader():
    def __init__(self, editor, model_type, flat_len=5, max_len=50):
        if model_type == 'spm':
            editor = editor.spm
        elif model_type == 'jieba':
            editor = editor.jieba
        else:
            raise ValueError('invalid model type')

        X, y = [], []
        for sent in editor['sents']:
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
                y.append(sent_ids[i + flat_len])
        self.X, self.y = np.asarray(X), to_categorical(y, editor['vocab_size'])
        self._y = y

if __name__ == '__main__':
    editor = editor.Editor('../data/result/segmented_01_spm.txt', '../data/result/segmented_01_jieba.txt', 5)
    dl_spm = DataLoader(editor, 'spm', 5, 50)
    dl_jieba = DataLoader(editor,'jieba', 5, 50)
    print(dl_spm.X, dl_spm.y, dl_spm._y)
    print('\n\n')
    print(dl_jieba.X, dl_jieba.y, dl_jieba._y)
