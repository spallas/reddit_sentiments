import pickle
import sys

from keras import Model, Input
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D
from imblearn.under_sampling import NearMiss

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ======== MODEL FLAGS ======== #

VOCAB_SIZE = 50_000
EMBED_SIZE = 300
LSTM_SIZE = 128

MAX_SENTENCE_LEN = 100

# ============================= #


def plot_heat_matrix(c_matrix, classes_strings):
    cm_dataframe = pd.DataFrame(c_matrix, index=classes_strings,
                                columns=classes_strings)
    plt.subplots(figsize=(10, 10))
    ax = sn.heatmap(cm_dataframe, cmap=plt.cm.jet, annot=True,
                    fmt="d", square=True)
    # turn the axis label
    for item in ax.get_yticklabels():
        item.set_rotation(0)

    for item in ax.get_xticklabels():
        item.set_rotation(90)

    plt.tight_layout()
    plt.show()


def load_dataset(data_file):
    with open(data_file, "rb") as f:
        X, Y, embed_mat = pickle.load(f)
    X = pad_sequences(X, maxlen=MAX_SENTENCE_LEN, truncating="post")
    Y = np.array(Y)
    nm1 = NearMiss(random_state=0, version=1)
    x_resampled, y_resampled = nm1.fit_sample(X, Y)
    x_train, x_dev, y_train, y_dev = train_test_split(x_resampled, y_resampled,
                                                      test_size=0.2, random_state=0)
    neg_sent_count = sum(Y)
    print("# neg:", neg_sent_count)
    return x_train, x_dev, y_train, y_dev, embed_mat

# ======== TRAINING ======== #


x_train, x_dev, y_train, y_dev, embeddings = load_dataset(sys.argv[1])

# classifier = svm.LinearSVC()
# classifier.fit(x_train, y_train)
# y_pred = classifier.predict(x_dev)
#
# print(classification_report(y_dev, y_pred, target_names=["non-burst", "burst"]))
# sys.exit(0) # gets to .66 F1 score...

input_size = MAX_SENTENCE_LEN

embedding_layer = Embedding(VOCAB_SIZE,
                            EMBED_SIZE,
                            weights=[embeddings],
                            input_length=input_size,
                            trainable=False)

input_layer = Input(shape=(input_size,), dtype='float32')

x = embedding_layer(input_layer)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Bidirectional(LSTM(LSTM_SIZE, dropout=0.2, recurrent_dropout=0.2))(x)

pred_layer = Dense(2, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=pred_layer)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=64,
          epochs=4,
          validation_data=(x_dev, y_dev))

model.save("reddit_model.h5")

# ======== EVALUATION ======== #

y_prob = model.predict(x_dev)
y_pred = y_prob.argmax(axis=-1)
plot_heat_matrix(confusion_matrix(y_dev, y_pred), ["non-burst", "burst"])
print(classification_report(y_dev, y_pred, target_names=["non-burst", "burst"]))
