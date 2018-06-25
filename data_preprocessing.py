import json
import pickle
import requests
import numpy as np
import mmap
import os
import sys
from tqdm import tqdm
import nltk
nltk.data.path.append("/Users/davidespallaccini/sourcecode/learning/nlp/nltk_data")
# nltk.download('punkt', download_dir="/Users/davidespallaccini/sourcecode/learning/nlp/nltk_data")

# ======== DATA FLAGS ======== #

MAX_SENTENCE_LEN = 100
GLOVE_FILE = "glove-51k.txt"
GLOVE_LIMIT = 50_000
EMBEDDINGS = "glove-50k.pkl"

DATASET_FILE = "reddit_data.pkl"

# ============================ #


def _get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def build_dataset(dataset_file):
    X = []
    Y = []

    word2id, id2word, embeddings_matrix = load_embeddings()

    print("Done.")
    counter = 0
    with open(dataset_file) as f:
        # show fancy progress bar
        for line in tqdm(f, total=_get_num_lines(dataset_file), desc=dataset_file):
            ids, label = line.split("\t")
            source_id = ids[2:-2].split("', '")[0]
            text = download_text(source_id)
            if text == "error":
                continue
            words_list = list(map(lambda x: word2id.get(x.lower(), 0),
                                  nltk.word_tokenize(text)))
            X.append(words_list)
            if label.strip() == "burst":
                Y.append(1)
            else:
                Y.append(0)

    print("Writing result to file...")
    with open(dataset_file+DATASET_FILE, "wb") as f:
        pickle.dump([X, Y, embeddings_matrix], f)

    return


def load_embeddings():
    if os.path.exists(EMBEDDINGS):
        with open(EMBEDDINGS, "rb") as f:
            return pickle.load(f)
    else:
        with open(EMBEDDINGS, "wb") as f:
            word2id = {"UNK": 0}
            id2word = {0: "UNK"}

            print("Loading embeddings...")
            glove = np.loadtxt(GLOVE_FILE, dtype='str', comments=None)
            words = glove[:GLOVE_LIMIT, 0]
            embeddings_matrix = glove[:GLOVE_LIMIT, 1:].astype('float')
            for w in words:
                id2word[len(id2word)] = w
                word2id[w] = len(id2word) - 1
            pickle.dump([word2id, id2word, embeddings_matrix], f)
            return word2id, id2word, embeddings_matrix


def download_text(post_id):
    posts_by_id = "https://api.pushshift.io/reddit/search/submission/?ids="
    comment_ids_by_post = "https://api.pushshift.io/reddit/submission/comment_ids/"
    comments_by_ids = "https://api.pushshift.io/reddit/search/comment/?ids="
    max_comments = 8
    max_chars = 512
    text = []

    try:
        resp = requests.get(posts_by_id + post_id + "&fields=title,selftext").json()
        text.append(resp["data"][0]["title"])
        text.append(resp["data"][0]["selftext"])
    except json.decoder.JSONDecodeError:
        return "error"

    try:
        resp = requests.get(comment_ids_by_post + post_id).json()
        comments = resp["data"]
        comments = comments[:max(len(comments), max_comments)]

        resp = requests.get(comments_by_ids + ",".join(comments) + "&fields=body").json()
        for r in resp["data"]:
            body = r["body"]
            body = body[:max(len(body), max_chars)]
            text.append(body)
    except json.decoder.JSONDecodeError:
        return " ".join(text)

    return " ".join(text)


def merge():
    X = []
    Y = []
    em = []
    for file_name in os.listdir(sys.argv[2]):
        with open(os.path.join(sys.argv[2], file_name), "rb") as f:
            x, y, em = pickle.load(f)
            X += x
            Y += y
    print("writing results...")
    with open(sys.argv[3], "wb") as f:
        pickle.dump([X, Y, em], f)


if sys.argv[1] == "merge":
    merge()
elif sys.argv[1] == "download":
    build_dataset(sys.argv[2])
else:
    print("Nope.")
