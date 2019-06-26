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
from utils import clean_text

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from utils import generate_features

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
    """
    Loads the data, if the data is not splitted yet the data will be split in a train and val set
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

        train = train[["tweet","subtask_a"]]
        val = val[["tweet","subtask_a"]]
        test = test[["tweet","subtask_a"]]

        train.columns = ['text', 'label']
        val.columns = ['text','label']
        test.columns = ['text', 'label']

        train.to_csv("data/train.csv", index=False)
        val.to_csv("data/val.csv", index=False)
        test.to_csv("data/test.csv", index=False)

    return train.head(10), val.head(10), test.head(10)

def load_data_features():
    """
    Loads the data, if the data is not splitted yet the data will be split in a train and val set
    """

    RANDOM_STATE = 123

    train_file = Path("data/train.csv")

    if train_file.exists():
        train = pd.read_csv("data/train.csv")
        val = pd.read_csv("data/val.csv")
        test = pd.read_csv("data/test.csv")

        features_train = pd.read_csv("data/features_train.csv")
        features_val = pd.read_csv("data/features_val.csv")
        features_test = pd.read_csv("data/features_test.csv")

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

        train = train[["tweet","subtask_a"]]; val = val[["tweet","subtask_a"]]; test = test[["tweet","subtask_a"]]

        train.columns = ['text', 'label']; val.columns = ['text','label']; test.columns = ['text', 'label']

        # Generate features
        features_train = generate_features(train)
        features_val = generate_features(val)
        features_test = generate_features(test)

        train.to_csv("data/train.csv", index=False)
        val.to_csv("data/val.csv", index=False)
        test.to_csv("data/test.csv", index=False)

        features_train.to_csv("data/features_train.csv")
        features_val.to_csv("data/features_val.csv")
        features_test.to_csv("data/features_test.csv")

    return train, val, test, features_train, features_val, features_test

#Clean Data
def clean_data(df, remove_punt_number_special_chars=False,remove_stopwords=False, apply_stemming=False):
    """Clean the data and remove data which has a length of less than 3 words
    Args:
        df: Dataframe
    """

    labels = encode_label(df["label"])
    text_clean = [clean_text(text, remove_punt_number_special_chars,remove_stopwords, apply_stemming) for text in df["text"]]

    df = pd.DataFrame({"text":text_clean, "label":labels})

    return text_clean, labels

#Get Dataloader
def get_dataloader_bert(examples, batch_size):
    """Make data iterator
        Arguments:
            X: Features
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

def get_dataloader(X, y, batch_size):

    """Make iterator for a given X and y and batch size
    Args:
        X: X vector
        y: y vector
        batch_size: (int) Batch size
    """

    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size)

    return loader

def get_dataloader_features(X, features, y, batch_size):

    """Make iterator for a given X and y and batch size
    Args:
        X: X vector
        y: y vector
        batch_size: (int) Batch size
    """

    X = torch.tensor(X, dtype=torch.long)
    print(X)
    print(features)
    features = torch.tensor(list(features), dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X, features, y)
    loader = DataLoader(ds, batch_size=batch_size)

    return loader

def encode_label(y):

    """Encode labels from str to numbers
    Args:
        y: y vector
    """

    y = y.values
    le = LabelEncoder()
    le.fit(y)

    return np.array(le.transform(y))

def get_data_bert(max_seq_length, batch_sizes):

    """
    Args:
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
    train_dataloader = get_dataloader_bert(train_examples, batch_sizes[0])
    val_dataloader = get_dataloader_bert(val_examples, batch_sizes[1])
    test_dataloader = get_dataloader_bert(test_examples, batch_sizes[2])

    return train_dataloader, val_dataloader, test_dataloader

def get_data(max_seq_len, embedding_file, batch_size):

    """
    Args:
        max_seq_len: Max sequence length of the sentences
        embedding_file: Embedding file
        batch_size: Batch size for the DataLoader

    Output:
        embedding_dim, word_index, embedding_matrix, X_train, y_train, X_test, y_test
    """

    #Load data
    train, val, test = load_data()

    #Embedding dimension based on the embedding_file
    embedding_dim = int(re.findall('\d{3,}', embedding_file)[0])

    #Clean data
    X_train, y_train = clean_data(train, remove_punt_number_special_chars=True,remove_stopwords=True, apply_stemming=False)
    X_val, y_val = clean_data(val, remove_punt_number_special_chars=True,remove_stopwords=True, apply_stemming=False)
    X_test, y_test = clean_data(train, remove_punt_number_special_chars=True,remove_stopwords=True, apply_stemming=False)

    tokenizer = Tokenizer(num_words = 10000000)
    tokenizer.fit_on_texts(list(X_train)+list(X_val))

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    #Embeddings
    embeddings_index = load_glove(embedding_file)
    embedding_matrix = create_weight_matrix(vocab_size, word_index, embedding_dim, embeddings_index)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=max_seq_len)

    X_val = tokenizer.texts_to_sequences(X_val)
    X_val = pad_sequences(X_val, maxlen=max_seq_len)

    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_seq_len)

    train_dataloader = get_dataloader(X_train, y_train, batch_size)
    val_dataloader = get_dataloader(X_val, y_val, batch_size)
    test_dataloader = get_dataloader(X_test, y_test, batch_size)

    return embedding_dim, int(vocab_size), embedding_matrix, train_dataloader, val_dataloader, test_dataloader

def get_data_features(max_seq_len, embedding_file, batch_size):

    """
    Args:
        max_seq_len: Max sequence length of the sentences
        embedding_file: Embedding file
        batch_size: Batch size for the DataLoader

    Output:
        embedding_dim, word_index, embedding_matrix, X_train, y_train, X_test, y_test
    """

    #Load data
    train, val, test, features_train, features_val, features_test = load_data_features()

    #Embedding dimension based on the embedding_file
    embedding_dim = int(re.findall('\d{3,}', embedding_file)[0])

    #Clean data
    X_train = [clean_text(text, remove_punt_number_special_chars=True,remove_stopwords=True, apply_stemming=False) for text in train["text"]]
    X_val = [clean_text(text, remove_punt_number_special_chars=True,remove_stopwords=True, apply_stemming=False) for text in val["text"]]
    X_test = [clean_text(text, remove_punt_number_special_chars=True,remove_stopwords=True, apply_stemming=False) for text in test["text"]]

    y_train = encode_label(train["label"])
    y_val = encode_label(val["label"])
    y_test = encode_label(test["label"])

    tokenizer = Tokenizer(num_words = 10000000)
    tokenizer.fit_on_texts(list(X_train)+list(X_val))

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    #Embeddings
    embeddings_index = load_glove(embedding_file)
    embedding_matrix = create_weight_matrix(vocab_size, word_index, embedding_dim, embeddings_index)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=max_seq_len)

    X_val = tokenizer.texts_to_sequences(X_val)
    X_val = pad_sequences(X_val, maxlen=max_seq_len)

    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_seq_len)

    train_dataloader = get_dataloader_features(X_train, features_train, y_train, batch_size)
    val_dataloader = get_dataloader_features(X_val, features_val, y_val, batch_size)
    test_dataloader = get_dataloader_features(X_test, features_test, y_test, batch_size)

    return embedding_dim, int(vocab_size), embedding_matrix, train_dataloader, val_dataloader, test_dataloader

def load_data_svm():

    #Load data
    train, val, test = load_data()

    #Clean data
    X_train = [clean_text(text, remove_punt_number_special_chars=True,remove_stopwords=True, apply_stemming=False) for text in train["text"]]
    X_val = [clean_text(text, remove_punt_number_special_chars=True,remove_stopwords=True, apply_stemming=False) for text in val["text"]]
    X_test = [clean_text(text, remove_punt_number_special_chars=True,remove_stopwords=True, apply_stemming=False) for text in test["text"]]

    y_train = encode_label(train["label"])
    y_val = encode_label(val["label"])
    y_test = encode_label(test["label"])

    bow_vec = CountVectorizer(ngram_range=(1,2), min_df=5, max_df=0.95, max_features=100000, stop_words='english')
    bow = bow_vec.fit(X_train)

    tfidf_vec = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.95, use_idf=True, smooth_idf=True, sublinear_tf=True, max_features=100000, stop_words='english')
    tfidf = tfidf_vec.fit(X_train)

    X_bow_train = tfidf_vec.transform(X_train)
    X_bow_test  = tfidf_vec.transform(X_test)

    X_tfidf_train = tfidf_vec.transform(X_train)
    X_tfidf_test  = tfidf_vec.transform(X_test)

    return X_bow_train, X_bow_test, X_tfidf_train, X_tfidf_test, y_train, y_test
