import sys
import pickle

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

from keras.models import load_model
from sklearn.model_selection import train_test_split

MAX_SENTENCE_LEN=100


def load_dataset(data_file):
    with open(data_file, "rb") as f:
        X, Y, _ = pickle.load(f)
    X = pad_sequences(X, maxlen=MAX_SENTENCE_LEN, truncating="post")
    Y = np.array(Y)
    _, _, y_train, y_dev = train_test_split(X, Y, random_state=1241251235)
    return y_train, y_dev


y_train, y_dev = load_dataset(sys.argv[1])

model = load_model(sys.argv[2])

y_prob = model.predict(x_dev)
y_pred = y_prob.argmax(axis=-1)

print(classification_report(y_dev, y_pred, target_names=["burst", "non-burst"]))
