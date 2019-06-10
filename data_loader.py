import json
from pathlib import Path
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
import torch

import re
import contractions
import emoji
import string
from nltk.tokenize import TweetTokenizer
from wordcloud import STOPWORDS
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from utils import convert_examples_to_features
from utils import clean_tweet

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_glove(embedding_file):

    """Load GloVe file
    Args:
        embedding_file: (str) Directory of the embedding file
    """

    EMBEDDING_FILE = embedding_file
    embeddings_index = dict()

    for line in open(EMBEDDING_FILE):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    print("Loaded {} word vectors".format(len(embeddings_index)))

    return embeddings_index

def create_weight_matrix(vocab_size, word_index, embedding_dim, embeddings_index):

    """Create weight matrix for the embeddings
    Args:
        vocab_size: Vocabulary size
        word_index: Word index
        embedding_dim: Dimension of the embeddings
        embeddings_index: Index of the embedding
    """
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())

            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


#Load data
def load_data():
    """Loads the data, if the data is not splitted yet the data will be split in a train and validation set
    """
    RANDOM_STATE = 123

    train_file = Path("data/train.csv")

    if train_file.exists():
        train = pd.read_csv("data/train.csv")
        val = pd.read_csv("data/val.csv")
        test = pd.read_csv("data/test.csv")
    else:
        # Split normal data
        train_cola = pd.read_csv("data/SemEval/olid-training-v1.0.tsv", delimiter="\t")
        test_cola = pd.read_csv("data/SemEval/testset-levela.tsv", delimiter="\t")
        labels_cola = pd.read_csv("data/SemEval/labels-levela.csv", header=None)
        labels_cola.columns = ['id', 'subtask_a']

        test = pd.merge(test_cola, labels_cola, on='id')

        # Remove duplicates
        train_cola = train_cola.drop_duplicates("tweet")
        test = test.drop_duplicates("tweet")

        train, val = train_test_split(train_cola, test_size=0.2, random_state=RANDOM_STATE)
        train.reset_index(drop=True)
        val.reset_index(drop=True)

        train.columns = ['text', 'label']
        val.columns = ['text','label']
        test.columns = ['text', 'label']

        train.to_csv("data/train.csv", index=False)
        val.to_csv("data/val.csv", index=False)
        test.to_csv("data/test.csv", index=False)

    return train, val, test

#Clean Data
def clean_data(df):
    """Clean the data and remove data which has a length of less than 3 words
    Args:
        df: Dataframe
    """
<<<<<<< HEAD
    labels = [0 if label=="NOT" else 1 for label in df["label"]]
    text_clean = [clean_text(text) for text in df["text"]]

    df = pd.DataFrame({"text":text_clean, "label":labels})

    length = [len(text.split(' ')) for text in df.text]
    df["length"] = length
    df = df[df["length"]<=3]
    df = df.drop(columns="length")
=======
    #labels = [0 if label=="NOT" else 1 for label in df["subtask_a"]]
    labels = encode_label(df["subtask_a"])
    #labels = encode_label(df["subtask_a"])

    tweet_clean = [clean_tweet(tweet) for tweet in df["tweet"]]

    # df = pd.DataFrame({"tweet":tweet_clean, "label":labels})
    #
    #length = [len(text.split(' ')) for text in tweet_clean]
    # df["length"] = length
    # df = df[df["length"]<=3]
    # df = df.drop(columns="length")
>>>>>>> 84c01cc1e8b8b4e17453aa409311c46544918b1e

    return tweet_clean, labels

#Get Dataloader
def get_dataloader(examples, batch_size):
    """Make data iterator
        Arguments:
            X:  Features
            y: Labels
            batch_size: (int) Batch size
    """

    all_input_ids = torch.tensor(list(examples.input_ids), dtype=torch.long)
    all_input_mask = torch.tensor(list(examples.input_mask), dtype=torch.long)
    all_segment_ids = torch.tensor(list(examples.segment_ids), dtype=torch.long)
    all_label_ids = torch.tensor(list(examples.label_id), dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

def make_iterator(X, y, batch_size):
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size)
    return loader

def encode_label(y):
    y = y.values
    le = LabelEncoder()
    le.fit(y)
    return np.array(le.transform(y))

def get_data_bert(max_seq_length, batch_sizes):

    """
    Arguments:
        max_num_words: (int) Max number of words as input for the Tokenizer
        embedding_dim: (int) Embedding dim of the embeddings
        max_seq_length: (int) Max sequence length of the sentences
        batch_size: (int) Batch size for the DataLoader
        use_bert: (bool) Use the BERT model or another model
    Output:
        word_index, embedding_matrix, X_train, y_train, X_test, y_test
    """

    #Load data
    train, val, test = load_data()

    #Clean data

    X_train, y_train = clean_data(train)
    X_val, y_val = clean_data(val)
    X_test, y_test = clean_data(test)

    #Features data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_examples = convert_examples_to_features(X_train, y_train, max_seq_length, tokenizer)
    val_examples = convert_examples_to_features(X_val, y_val, max_seq_length, tokenizer)
    test_examples = convert_examples_to_features(X_test, y_test, max_seq_length, tokenizer)

    #Data loaders
    train_dataloader = get_dataloader(train_examples, batch_sizes[0])
    val_dataloader = get_dataloader(val_examples, batch_sizes[1])
    test_dataloader = get_dataloader(test_examples, batch_sizes[2])

    return train_dataloader, val_dataloader, test_dataloader
<<<<<<< HEAD
=======

def get_data(max_seq_len, embedding_file, batch_size):

    """
    Arguments:
        max_seq_len: Max sequence length of the sentences
        batch_size: Batch size for the DataLoader

    Output:
        word_index, embedding_matrix, X_train, y_train, X_test, y_test
    """
    #Load data
    train, val, test = load_data()

    #Embedding dimension based on the embedding_file
    embedding_dim = int(re.findall('\d{3,}', embedding_file)[0])

    #Clean data
    X_train, y_train = clean_data(train)
    X_val, y_val = clean_data(val)
    X_test, y_test = clean_data(test)

    tokenizer = Tokenizer(num_words = 10000000)
    tokenizer.fit_on_texts(list(X_train))

    vocab_size = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index

    #Embeddings
    embeddings_index = load_glove(embedding_file)
    embedding_matrix = create_weight_matrix(vocab_size, word_index, embedding_dim, embeddings_index)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=max_seq_len)

    X_val = tokenizer.texts_to_sequences(X_val)
    X_val = pad_sequences(X_val, maxlen=max_seq_len)

    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_seq_len)

    train_dataloader = make_iterator(X_train, y_train, batch_size)
    val_dataloader = make_iterator(X_val, y_val, batch_size)
    test_dataloader = make_iterator(X_test, y_test, batch_size)

    return int(vocab_size), embedding_matrix, train_dataloader, val_dataloader, test_dataloader
>>>>>>> 84c01cc1e8b8b4e17453aa409311c46544918b1e
