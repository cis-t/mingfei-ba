# -*- coding: utf-8 -*-
""" This file shows how sentencepiece can be used to segment a given texts. """
import re
import sentencepiece as spm


def segment_with_spm(inp_, out_, model_name, vocab_size):
    """ Given a txt file, train a spm model.
    inp_: path of your input txt file.
    out_: path of your out txt file -- segmented texts.
    """
    # training an spm model use the input txt file
    # note the vocabulary size is a hyper-parameter
    spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={}'.format(inp_, model_name, vocab_size))
    
    # after training, apply the trained model to the input txt file
    # for segmentation
    sp = spm.SentencePieceProcessor()
    sp.Load('./{}.model'.format(model_name))  # load our trained model
    
    with open(inp_, 'r') as file:
        # read in the unsegmented lines
        result = re.sub(r' \—\"\—|世 界 文 学 名 著 百 部|红 与 黑|\,|\—\!\—|\—\!\"\—|\—\!(.)\!\—| \—\!\"\#\—|\—\!\"\#\—| \—\"\#\$\—| \—\!\!\"\—|\—\"\#\"\—|\—\%\&\’\—|" "|\—.*\—', '', file.read())
        result = re.sub(r'\n\n\n*','', result)
        verses = re.split(r'[?。! ; ,]', result)

        #verses = [l.strip() for l in verses]
        
        # use the loaded model sp to segment the unsegmented lines
    segmented_verses = [sp.EncodeAsPieces(v) for v in verses]
    assert len(segmented_verses) == len(verses)

    # write out segmented results
    with open(out_, 'w') as f:
        for l in segmented_verses:
            f.write("{}\n".format(" ".join(l)))

    print("Segmented file written to {}".format(out_))




if __name__ == "__main__":
    segment_with_spm(
        inp_="../data/test/01.txt",
        model_name="spm",
        out_="../data/result/segmented_01_spm.txt",
        vocab_size=20000          #max
    )
    segment_with_spm(
        inp_="../data/test/02.txt",
        model_name="spm",
        out_="../data/result/segmented_02_spm.txt",
        vocab_size=20000  # max
    )