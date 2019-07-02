from keras.utils import to_categorical
from keras.utils.data_utils import get_file
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

from collections import Counter
import itertools as its
import numpy as np


class BibleEdition():
	def __init__(self, txt_, min_count=5):
		self.sents = self.clean_lines(txt_)
		w2c = Counter(" ".join(its.chain.from_iterable(self.sents)).split(" ")).most_common()
		w2i = {}
		for idx, (w, c) in enumerate(w2c):
			if c < min_count:
				break
			w2i[w] = idx
		w2i["<unk>"] = len(w2i)
		w2i["</s>"] = len(w2i)
		i2w = {v: k for k, v in w2i.items()}
		self.w2i, self.i2w = w2i, i2w
		self.vocab_size = len(self.w2i)
	
	@classmethod
	def clean_lines(cls, txt_):
		sents = []
		with open(txt_, "r") as f:
			for l in f:
				if l[0] == "#":
					continue
				l = l.strip().split("\t")[-1].split(" ")
				if len(l) == 0:
					continue
				sents.append(l)
		print("Obtained {} verses from {}.".format(
			len(sents), txt_))
		return sents


class DataLoader(object):
	def __init__(self, edition, flat_len=5, max_len=50):
		X, y = [], []
		for sent in edition.sents:
			if len(sent) < max_len:
				sent += ["</s>"] * (max_len - len(sent))
			else:
				sent = sent[:max_len-1] + ["</s>"]
			sent_ids = []
			for w in sent:
				try:
					sent_ids.append(edition.w2i[w])
				except KeyError:
					sent_ids.append(edition.w2i["<unk>"])
			for i in range(0, len(sent_ids) - flat_len, flat_len):
				X.append(sent_ids[i:i+flat_len])
				y.append(sent_ids[i+flat_len])
		self.X, self.y = np.asarray(X), to_categorical(y, edition.vocab_size)
		self._y = y


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

	edition = BibleEdition("zho.txt", min_count=MIN_COUNT)
	dl = DataLoader(edition, max_len=MAX_LEN)
	lm_model = get_lm_model(vocab_size=edition.vocab_size,flat_len=FLAT_LEN,emb_dim=EMB_DIM,lstm_dim=LSTM_DIM)
	
	NUM_EPOCH = 100
	BATCH_SIZE = 1024
	VAL_SPLIT = 0.2
	SAVED_PATH = "saved_model.pkl"

	# evaluted
	inference = True

	if not inference:
		stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="auto")
		saver = ModelCheckpoint(SAVED_PATH, monitor="val_loss", verbose=0, save_best_only=True)
		history = lm_model.fit(dl.X, dl.y, batch_size=BATCH_SIZE,epochs=NUM_EPOCH,verbose=1,validation_split=VAL_SPLIT,callbacks=[stop, saver])
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