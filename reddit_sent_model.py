import pickle
import sys
import numpy as np

import tensorflow as tf

from keras import Model, Input
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, Dropout, LSTM, Bidirectional, Flatten

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
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
    ax = sn.heatmap(cm_dataframe, cmap=plt.cm.jet, annot=False, square=True)
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
    x_train, x_dev, y_train, y_dev = train_test_split(X, Y, random_state=1241251235)
    return x_train, x_dev, y_train, y_dev, embed_mat

# ======== TRAINING ======== #


x_train, x_dev, y_train, y_dev, embeddings = load_dataset(sys.argv[1])
y_dev[0] = 1

input_size = MAX_SENTENCE_LEN

embedding_layer = Embedding(VOCAB_SIZE,
                            EMBED_SIZE,
                            weights=[embeddings],
                            input_length=input_size,
                            trainable=False)

input_layer = Input(shape=(input_size,), dtype='float32')
embeddings_output = embedding_layer(input_layer)

x = Bidirectional(LSTM(LSTM_SIZE, dropout=0.2, recurrent_dropout=0.2))(embeddings_output)

pred_layer = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_layer, outputs=pred_layer)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print(x_train.shape)
print(x_train[0])
print(x_dev.shape)
print(y_train.shape)
print(y_dev.shape)

model.fit(x_train, y_train,
          batch_size=16,
          epochs=1,
          validation_data=(x_dev, y_dev))

model.save("reddit_model.h5")

# ======== EVALUATION ======== #

y_prob = model.predict(x_dev)
y_pred = y_prob.argmax(axis=-1)
y_pred[1] = 1
plot_heat_matrix(confusion_matrix(y_dev, y_pred), ["burst", "non-burst"])
print(classification_report(y_dev, y_pred, target_names=["burst", "non-burst"]))
