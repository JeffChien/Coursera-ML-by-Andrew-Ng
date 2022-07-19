import pathlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string


#%%
def get_vocab_list(filename):
    data = np.genfromtxt(
        "../../data/vocab.txt",
        dtype=[("id", "i8"), ("voc", "U256")],
        delimiter="\t",
    )
    vocab_list = data["voc"]
    vocab_pos = {v: i for i, v in enumerate(vocab_list)}
    return vocab_list, vocab_pos


#%%
def read_file(filename):
    content = ""
    with open(filename, "r", encoding="ISO-8859-1") as f:
        content = f.read()
    return content


#%%
pattern_html = re.compile(r"<[^<>]+>")
pattern_number = re.compile(r"[0-9]+")
pattern_urls = re.compile(r"(http|https)://[^\s]*")
pattern_email = re.compile(r"[^\s]+@[^\s]+")
pattern_dolar = re.compile(r"[$]+")
pattern_alphanum_only = re.compile(r"[^a-zA-Z0-9]")


def process_email(email_contents: str, vocab_pos):
    word_indices = []

    stemmer = PorterStemmer()
    email_contents = email_contents.lower()

    email_contents = pattern_html.sub(" ", email_contents)
    email_contents = pattern_number.sub("number", email_contents)
    email_contents = pattern_urls.sub("httpaddr", email_contents)
    email_contents = pattern_email.sub("emailaddr", email_contents)
    email_contents = pattern_dolar.sub("dollar", email_contents)

    for word in word_tokenize(
        email_contents.translate(str.maketrans("", "", string.punctuation))
    ):
        word = pattern_alphanum_only.sub("", word)
        word = stemmer.stem(word)
        if len(word) < 1:
            continue

        if word in vocab_pos:
            word_indices.append(vocab_pos[word])

    return word_indices


#%%
def email_features(word_indices, N: int):
    x = np.zeros(N)
    x[word_indices] = 1
    return x


def load_email_data(dirname, feature_pos, label_as, m: int = -1):
    n = len(feature_pos)
    dirpath = pathlib.Path(dirname)
    flist = [p for p in dirpath.iterdir() if p.is_file()]
    if m > -1:
        flist = flist[:m]

    X = np.empty((0, n))
    y = np.zeros(0)

    for i, filename in enumerate(flist):
        file_content = read_file(filename)
        word_indices = process_email(file_content, feature_pos)
        x = email_features(word_indices, n)
        X = np.r_[X, [x]]
        y = np.hstack((y, label_as))
    return X, y
