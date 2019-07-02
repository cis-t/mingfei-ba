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
OPTIMIZER = "adam"
ACTIVATION = "softmax"
METRICS = ["categorical_accuracy"]
#MAX_LEN = 20


def get_lm_model(vocab_size, flat_len, emb_dim, lstm_dim):
    model = Sequential()
    model.add(Embedding(vocab_size, emb_dim, input_length=flat_len))
    model.add(LSTM(lstm_dim))
    model.add(Dense(vocab_size, activation=ACTIVATION))
    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
    print(model.summary())
    return model

def trainModel(editor, model_type, saved_path, notYetTrained=True):
    dl = DataLoader(editor, model_type)
    if model_type == 'mod_spm':
        editor = editor.mod_spm
    elif model_type == 'mod_jieba':
        editor = editor.mod_jieba
    elif model_type == 'cls_spm':
        editor = editor.cls_jieba
    elif model_type == 'cls_jieba':
        editor = editor.cls_jieba
    else:
        raise ValueError('invalid model type')
    
    lm_model = get_lm_model(
        vocab_size=editor['vocab_size'],
        flat_len=FLAT_LEN,
        emb_dim=EMB_DIM,
        lstm_dim=LSTM_DIM)

    NUM_EPOCH = 50 # 100
    BATCH_SIZE = 1024
    SAVED_PATH = saved_path

    if notYetTrained:
        # vermeidung von Overfitting
        stop = EarlyStopping(
            monitor="val_loss", min_delta=0,
            patience=5, verbose=1, mode="auto")
        saver = ModelCheckpoint(
            SAVED_PATH, monitor="val_loss",
            verbose=0, save_best_only=True)
        lm_model.fit(
            dl.train_X, dl.train_Y,
            validation_split=0.2,

            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCH,
            verbose=1,
            callbacks=[stop, saver])
    else:
        lm_model = load_model(SAVED_PATH)
        #todo: evaluate
        loss, accuracy = lm_model.evaluate(dl.test_X, dl.test_Y)
        print('LOSS:', loss)
        print('ACCURACY:', accuracy)

        gold_tokens, pred_tokens = [], []
        for step in range(1000):
            inp = dl.test_X[step]
            pred = np.argmax(lm_model.predict(np.asarray([inp])))
            gold_tokens.append(editor['i2w'][dl.test_original_Y[step]])
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
#    MAX_LEN = 50
    FLAT_LEN = 5
    MIN_COUNT = 5
    EMB_DIM = 100
    LSTM_DIM = 100

    editor = Editor('../data/result/segmented_01_spm.txt', '../data/result/segmented_01_jieba.txt','../data/result/segmented_02_spm.txt', '../data/result/segmented_02_jieba.txt', MIN_COUNT)
    editor_mod_spm = editor.mod_spm
    editor_mod_jieba = editor.mod_jieba
#   editor_cls_spm = editor.cls_spm
#    editor_cls_jieba = editor.cls_jieba
#

    print('Modern Chinese SPM')
    trainModel(editor, 'mod_spm', 'saved_mod_spm.pkl', True)
    trainModel(editor, 'mod_spm', 'saved_mod_spm.pkl', False)
    print('Modern Chinese Jieba')
    trainModel(editor, 'mod_jieba', 'saved_mod_jb.pkl',True)
    trainModel(editor, 'mod_jieba', 'saved_mod_jb.pkl',False)

#    print('Classic Chinese SPM')
#    trainModel(editor, 'cls_spm', 'saved_cls_spm.pkl', True)
#    trainModel(editor, 'cls_spm', 'saved_cls_spm.pkl', False)
#    print('Classic Chinese Jieba')
#    trainModel(editor, 'cls_jieba', 'saved_cls_jb.pkl',True)
#    trainModel(editor, 'cls_jieba', 'saved_cls_jb.pkl',False)


