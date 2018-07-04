import sys
import pickle

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

from keras.models import load_model

MAX_SENTENCE_LEN = 100


def load_test(data_file):
    with open(data_file, "rb") as f:
        X, Y = pickle.load(f)
    X = pad_sequences(X, maxlen=MAX_SENTENCE_LEN, truncating="post")
    Y = np.array(Y)
    return X, Y


x_test, y_test = load_test(sys.argv[1])

model = load_model(sys.argv[2])

y_prob = model.predict(x_test)
y_pred = y_prob.argmax(axis=-1)

print(classification_report(y_test, y_pred, target_names=["burst", "non-burst"]))
