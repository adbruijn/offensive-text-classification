import pandas as pd
import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report

import json
import logging
import os
import shutil

import torch

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

#Metrics
def accuracy_recall_precision_f1(y_pred, y_target):

    """Computes the accuracy, recall, precision and f1 score for given predictions and targets
    Args:
        y_pred: Logits of the predictions for each class
        y_target: Target values
    """

    y_pred = y_pred.cpu()
    y_target = y_target.cpu().numpy()

    predictions = torch.argmax(y_pred, dim=1).detach().numpy()

    correct = np.sum(predictions == y_target)
    accuracy = correct / len(predictions)

    accuracy = correct / len(predictions)

    recall = recall_score(y_target, predictions, average=None)
    precision = precision_score(y_target, predictions, average=None)
    f1 = f1_score(y_target, predictions, average=None)

    return accuracy, recall, precision, f1

#Preprocess Data
def clean_tweet(tweet, remove_punt_number_special_chars=False,remove_stopwords=False, apply_stemming=False):
    """Clean tweets
    Args:
        tweet: (str) Tweet
        remove_punt_number_special_chars: (bool) Remove punctuations, numbers and special characters
        remove_stopwords: (bool) Remove stopwords
        apply_stemming: (bool) Apply stemming on the words on the tweets
    """
    #Remove emojis
    tweet = re.sub(":[a-zA-Z\-\_]*:","", emoji.demojize(tweet)) #:hear-no-evil_monkey:
    tweet = re.sub(":\w+:","", emoji.demojize(tweet))
    tweet = re.sub(":\w+\’\w+:","", emoji.demojize(tweet)) #:woman's_boot:

    #Remove mentions, usernames (@USER)
    tweet = re.sub("\s*@USER\s*", '', tweet)

    #Remove URL
    tweet = re.sub("\s*URL\s*", '', tweet)

    #And
    tweet = re.sub("&amp;", "and", tweet)
    tweet = re.sub("&lt;", "<", tweet)
    tweet = re.sub("&gt", ">", tweet)
    tweet = re.sub("&", "and", tweet)

    #Replace contractions and slang of word
    tweet = re.sub("i'm", "I'm", tweet)
    tweet = contractions.fix(tweet, slang=True)

    #Lowercase
    tweet = tweet.lower()

    #Remove Hashtags + Words
    tweet = re.sub("#\s*\w+\s*", '', tweet)

    #Remove repeating whitespaces
    tweet = re.sub("\s[2, ]"," ", tweet)

    #Remove non ascii characters
    tweet.encode("ascii", errors="ignore").decode()

    #Remove punctuations, numbers and special characters (remove emoticons)
    if remove_punt_number_special_chars:
        tweet = re.sub('[^a-zA-Z]', ' ', tweet)

    #Tokenize tweet
    tt = TweetTokenizer(preserve_case=False,
                    strip_handles=True,
                    reduce_len=True)

    tweet_tokens = tt.tokenize(tweet)

    #Remove stopwords
    if remove_stopwords:
        stopwords = set(STOPWORDS)
        tweet_tokens = [token for token in tweet_tokens if token not in stopwords]

    #Stemming
    if apply_stemming:
        tweet_stem = [stemmer.stem(token) for token in tweet_tokens]

    clean = " ".join(tweet_tokens)

    return clean

#BERT
def convert_examples_to_features(data, max_seq_length, tokenizer):

    """Loads a data file and returns examples (input_ids, input_mask, segment_ids, label_id).
    Args:
        data: Data
        max_seq_length: (int) Maximum length of the sequences
        tokenizer: Tokenizer
    """

    col_names = ["input_ids","input_mask","segment_ids","label_id"]
    features = pd.DataFrame(columns=col_names)

    for index, example in data.iterrows():

        tokens_tweet = tokenizer.tokenize(example.tweet)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_tweet) > max_seq_length - 2:
            tokens_tweet = tokens_tweet[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_tweet + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        label_id = int(example.label)

        input_features = {'input_ids': input_ids, 'input_mask':input_mask, 'segment_ids':segment_ids, 'label_id':label_id}
        features.loc[len(features)] = input_features

    return features

#Checkpoints save and load
def save_checkpoint(state, directory, checkpoint):

    """Saves model and training parameters at checkpoint
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    """

    filepath = directory + checkpoint

    if not os.path.exists(directory):
        os.mkdir(directory)

    torch.save(state, filepath)


def load_checkpoint(checkpoint, model, optimizer=None):

    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """

    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
