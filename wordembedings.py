import numpy as np
import pandas as pd
from tqdm import tqdm

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from data_loader import load_data

from gensim.models import KeyedVectors

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):

    """Load embeddings file
    Args:
        embedding_file: (str) Directory of the embedding file
    """
    embeddings_index = dict(get_coefs(*line.split(" ")) for line in open(path))

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

def load_embedding_matrix(vocab_size, word_index, embedding_file):

    embedding_type = re.findall('\w+', embedding_file)[1]

    if (embedding_type == 'GloVe' or embedding_type == 'fastText'):
        embedding_dim = int(re.findall('\d{3,}', embedding_file)[0])

        embeddings_index = load_embeddings(embedding_file)
        print("Loaded {} word vectors".format(len(embeddings_index)))

    elif embedding_type == 'Word2Vec':
        embedding_dim = 300
        word2vec_dict = KeyedVectors.load_word2vec_format(embedding_file, binary=True)

        embeddings_index = dict()

        for word in word2vec_dict.wv.vocab:
            embeddings_index[word] = word2vec_dict.word_vec(word)

        print("Loaded {} word vectors".format(len(embeddings_index)))

    embedding_matrix = create_weight_matrix(vocab_size, word_index, embedding_dim, embeddings_index)

    return embedding_matrix

embedding_file_glove = 'data/GloVe/glove.twitter.27B.200d.txt'
embedding_file_word2vec = 'data/Word2Vec/GoogleNews-vectors-negative300.bin'
embedding_file_fasttext = 'data/fastText/wiki-news-300d-1M.vec'

#Load data
import re
from data_loader import clean_data
train, val, test = load_data('a')
max_features = 10000000
maxlen = 60

#Cleaning
X_train, y_train = clean_data(train, remove_punt_number_special_chars=True,remove_stopwords=True, apply_stemming=False)

#Tokenize the tweets
tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(list(X_train))

X_train = tokenizer.texts_to_sequences(X_train)

#Pad the tweets
X_train = pad_sequences(X_train, maxlen=maxlen)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

embedding_matrix = load_embedding_matrix(vocab_size, word_index, embedding_file_fasttext)
print(embedding_matrix)
