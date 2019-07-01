import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from keras import optimizers
from keras.utils import to_categorical
from keras.utils.data_utils import get_file
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.training_utils import multi_gpu_model

from collections import Counter
import itertools as its
import numpy as np


class BibleEdition():
	def __init__(self, txt_, min_count=5):
		sent = " ".join(its.chain.from_iterable(self.clean_lines(txt_))).split(" ")
		w2c = Counter(sent).most_common()
		w2i = {}
		for idx, (w, c) in enumerate(w2c):
			if c < min_count:
				break
			w2i[w] = idx
		w2i["<unk>"] = len(w2i)
		i2w = {v: k for k, v in w2i.items()}
		self.words = [t if t in w2i else "<unk>" for t in sent]
		self.ids = [w2i[w] for w in self.words]
		self.w2i, self.i2w = w2i, i2w
		self.vocab_size = len(self.w2i)
	
	@classmethod
	def clean_lines(cls, txt_):
		sents = []
		with open(txt_, "r") as f:
			for l in f:
				if l[0] == "#":
					continue
				l = l.lower()
				l = l.strip().split("\t")[-1].split(" ")
				if len(l) == 0:
					continue
				sents.append(l)
		print("Obtained {} verses from {}.".format(
			len(sents), txt_))
		return sents


class DataLoader(object):
	def __init__(self, edition, flat_len):
		X, y = [], []
		for i in range(len(edition.words)-flat_len):
			X.append(edition.ids[i:i+flat_len])
			y.append(edition.ids[i+flat_len])
		self.X, self.y = np.asarray(X), to_categorical(y, edition.vocab_size)
		self.decimal_y = y


def get_lm_model(vocab_size, flat_len, emb_dim, lstm_dim):
	model = Sequential()
	model.add(Embedding(vocab_size, emb_dim, input_length=flat_len))
	model.add(LSTM(lstm_dim))
	model.add(Dense(vocab_size, activation="softmax"))
	opt = optimizers.SGD(lr=0.001, clipnorm=1.)
	model.compile(loss="categorical_crossentropy", optimizer=opt)
	print(model.summary())
	return model


if __name__ == "__main__":
	FLAT_LEN = 50
	MIN_COUNT = 1
	EMB_DIM = 2048
	LSTM_DIM = 2048

	edition = BibleEdition("eng.txt", min_count=MIN_COUNT)
	dl = DataLoader(edition, flat_len=FLAT_LEN)
	lm_model = get_lm_model(vocab_size=edition.vocab_size, flat_len=FLAT_LEN, emb_dim=EMB_DIM, lstm_dim=LSTM_DIM)
	
	NUM_EPOCH = 10
	BATCH_SIZE = 128
	VAL_SPLIT = 0.2
	SAVED_PATH = "saved_model.pkl"

	inference = False

	if not inference:
		stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="auto")
		saver = ModelCheckpoint(SAVED_PATH, monitor="val_loss", verbose=0, save_best_only=True)
		history = lm_model.fit(dl.X, dl.y, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1, validation_split=VAL_SPLIT, callbacks=[stop, saver])
	else:
		lm_model = load_model(SAVED_PATH)
		gold_tokens, pred_tokens = [], []
		for step in range(10000):
			inp = dl.X[step]
			pred = np.argmax(lm_model.predict(np.asarray([inp])))
			pred_tokens.append(edition.i2w[pred])
			gold_tokens.append(edition.i2w[dl.decimal_y[step]])
			if step % 50 == 0:
				print("pred:")
				print(" ".join(pred_tokens))
				print("gold:")
				print(" ".join(gold_tokens))
				print("=" * 20)
				gold_tokens, pred_tokens = [], []
