from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from LM.Dataloader import DataLoader
from LM.Editor import Editor
from sklearn.metrics import accuracy_score

from keras.metrics import categorical_accuracy
from keras.preprocessing import sequence

LOSS_FUNCTION = "categorical_crossentropy"
OPTIMIZER ="adam"
ACTIVATION = "softmax"


def get_lm_model(vocab_size, flat_len, emb_dim, lstm_dim):
    model = Sequential()
    model.add(Embedding(vocab_size, emb_dim, input_length=flat_len))
    model.add(LSTM(lstm_dim))
    model.add(Dense(vocab_size, activation=ACTIVATION))
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER)
    print(model.summary())
    return model

def trainModel(editor, model_type, saved_path, notYetTrained=True):
    dl = DataLoader(editor, model_type, max_len=MAX_LEN)
    if model_type == 'spm':
        editor = editor.spm
    elif model_type == 'jieba':
        editor = editor.jieba
    else:
        raise ValueError('invalid model type')
    
    lm_model = get_lm_model(
        vocab_size=editor['vocab_size'],
        flat_len=FLAT_LEN,
        emb_dim=EMB_DIM,
        lstm_dim=LSTM_DIM)

    NUM_EPOCH = 50 # 100
    BATCH_SIZE = 1024
    VAL_SPLIT = 0.2
    SAVED_PATH = saved_path

    if not notYetTrained:
        # vermeidung von Overfitting
        stop = EarlyStopping(
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

        #lm_model.evaluate()

        gold_tokens, pred_tokens = [], []
        for step in range(1000):
            inp = dl.X[step]
            pred = np.argmax(lm_model.predict(np.asarray([inp])))
            gold_tokens.append(editor['i2w'][dl._y[step]])
            pred_tokens.append(editor['i2w'][pred])
            if step % 50 == 0:
                print("gold:")
                print(" ".join(gold_tokens))
                print("pred:")
                print(" ".join(pred_tokens))
                print("=" * 20)
                print(accuracy_score(gold_tokens, pred_tokens))

                #print(categorical_accuracy(gold_tokens, pred_tokens))

                gold_tokens, pred_tokens = [], []


if __name__ == "__main__":
    MAX_LEN = 50
    FLAT_LEN = 5
    MIN_COUNT = 5
    EMB_DIM = 100
    LSTM_DIM = 100

    editor = Editor('../data/result/segmented_01_spm.txt', '../data/result/segmented_01_jieba.txt', MIN_COUNT)


    editor_spm = editor.spm
    editor_jieba = editor.jieba

    print('SPM')
    trainModel(editor, 'spm', 'saved_spm.pkl', True)
    trainModel(editor, 'spm', 'saved_spm.pkl', False)
    print('Jieba')
    trainModel(editor, 'jieba', 'saved_jb.pkl',True)
    trainModel(editor, 'jieba', 'saved_jb.pkl',False)



  #  trainModel(editor_jieba, 'saved_jieba.pkl')

