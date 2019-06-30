from collections import Counter
import itertools as its

class Editor():
    def __init__(self, txt_spm, txt_jieba, min_count=5):
        self.spm = self.load(txt_spm, min_count, 'spm')
        self.jieba = self.load(txt_jieba, min_count, 'jieba')

    def load(self, txt_, min_count, model_type):
        model = {}
        sents = self.clean_lines(txt_, model_type)

        split_at_1 = round(len(sents) * 0.6)
        split_at_2 = round(len(sents) * 0.8)
        training = sents[:split_at_1]
        dev = sents[split_at_1:split_at_2]
        test = sents[split_at_2:]

        w2c = Counter(" ".join(its.chain.from_iterable(training)).split(" ")).most_common()
        w2i = {}
        for idx, (w, c) in enumerate(w2c):
            if c < min_count:
                break
            w2i[w] = idx
        w2i["<unk>"] = len(w2i)
        w2i["</s>"] = len(w2i)
        i2w = {v: k for k, v in w2i.items()}

        model['sents'] = training
        model['w2i'] = w2i
        model['i2w'] = i2w
        model['vocab_size'] = len(w2i)
        return model, dev, test

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

if __name__ == '__main__':
    editor = Editor('../data/result/segmented_01_spm.txt', '../data/result/segmented_01_jieba.txt', 5)
    spm = editor.spm['vocab_size']
    jieba = editor.jieba['vocab_size']
    print('SPM sentences:', spm, '\nJieba sentences:', jieba)

