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

#Preprocess Data
def clean_bert(tweet):

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

    #Remove repeating characters (whitespace)
    tweet = re.sub("\s[2, ]"," ", tweet)

    #Remove non ascii characters
    tweet.encode("ascii", errors="ignore").decode()

    #Tokenize tweet
    tt = TweetTokenizer(preserve_case=False,
                    strip_handles=True,
                    reduce_len=True)

    tweet_tokens = tt.tokenize(tweet)

    clean = " ".join(tweet_tokens)

    return clean

#Load data
def load_data():
    RANDOM_STATE = 123

    # Split normal data
    train_cola = pd.read_csv("data/SemEval/olid-training-v1.0.tsv", delimiter="\t")
    test_cola = pd.read_csv("data/SemEval/testset-levela.tsv", delimiter="\t")
    labels_cola = pd.read_csv("data/SemEval/labels-levela.csv", header=None)
    labels_cola.columns = ['id', 'subtask_a']

    test = pd.merge(test_cola, labels_cola, on='id')

    # Remove duplicates
    train_cola = train_cola.drop_duplicates("tweet")
    test = test.drop_duplicates("tweet")

    train_cola.to_csv("./data/SemEval/train_no_val.csv", index=False)

    train, val = train_test_split(train_cola, test_size=0.2, random_state=RANDOM_STATE)
    train.reset_index(drop=True)
    val.reset_index(drop=True)

    return train, val, test

#Clean Data
def clean_data(df):
    labels = [0 if label=="NOT" else 1 for label in df["subtask_a"]]
    tweet_clean = [clean_bert(tweet) for tweet in df["tweet"]]

    df = pd.DataFrame({"tweet":tweet_clean, "label":labels})

    length = [len(text.split(' ')) for text in df.tweet]

    df["length"] = length

    df = df[df["length"]<=3]

    df = df.drop(columns="length")

    return df

#Get Dataloader
def get_dataloader(examples, batch_size):

    all_input_ids = torch.tensor(list(examples.input_ids), dtype=torch.long)
    all_input_mask = torch.tensor(list(examples.input_mask), dtype=torch.long)
    all_segment_ids = torch.tensor(list(examples.segment_ids), dtype=torch.long)
    all_label_ids = torch.tensor(list(examples.label_id), dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader
