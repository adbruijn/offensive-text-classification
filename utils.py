import pandas as pd
import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report, confusion_matrix

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

from nltk.stem.porter import *
stemmer = PorterStemmer()

from nltk.corpus import stopwords
import emoji
import string
from textblob import TextBlob

stopwords = stopwords.words('english')

#Features
def generate_features(df):

    features = pd.DataFrame({'text':df['text']})

    features['emojis'] = features["text"].apply(lambda x: emoji.emoji_count(x))
    features['urls'] = features["text"].apply(lambda x: len(re.findall("URL", str(x))))
    features['hashtags'] = features["text"].apply(lambda x: len(re.findall("#", str(x))))
    features['users'] = features["text"].apply(lambda x: len(re.findall("USER", str(x))))

    features['words'] = features['text'].apply(lambda x: len(str(x).split(" ")))
    features['unique_words'] = features['text'].apply(lambda x: len(set(str(x).split(" "))))
    features['chars'] = features['text'].str.len()
    features['stopwords'] = features['text'].apply(lambda x: len([x for x in x.split() if x in stopwords]))
    features['punctuations'] = features["text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))-features['users']-features['hashtags']
    features['numerics'] = features['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

    features['upper'] = features['text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))-features['urls']-features['users']
    features['title'] = features['text'].apply(lambda x: len([x for x in x.split() if x.istitle()]))

    #features['polarity'] = features['text'].apply(lambda x: np.round(TextBlob(x).sentiment.polarity, 2))
    #features['subjectivity'] = features['text'].apply(lambda x: np.round(TextBlob(x).sentiment.subjectivity, 2))

    features = features.drop(columns='text', axis=1)

    return features

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
    #predictions = torch.max(y_pred, 1)[1].view(y_target.size()).data

    correct = np.sum(predictions == y_target)
    accuracy = correct / len(predictions)

    recall = recall_score(y_target, predictions, average=None) #average=None (the scores for each class are returned)
    precision = precision_score(y_target, predictions, average=None)
    f1 = f1_score(y_target, predictions, average=None)

    return accuracy, recall, precision, f1

def calculate_confusion_matrix(y_pred, y_target):

    y_pred = y_pred.cpu()
    y_target = y_target.cpu().numpy()

    predictions = torch.argmax(y_pred, dim=1).detach().numpy()

    #Confusion matrix
    cm = confusion_matrix(y_target, predictions)

    #multi_cm = multilabel_confusion_matrix(y_target, predictions)
    #print(multi_cm)
    #print(confusion_matrix(y_target, predictions))

    #Classification report
    #print(classification_report(y_target, predictions))

    return cm

#Preprocess Data
def clean_text(text, remove_punt_number_special_chars=False,remove_stopwords=False, apply_stemming=False):
    """Clean text
    Args:
        text: (str) Text
        remove_punt_number_special_chars: (bool) Remove punctuations, numbers and special characters
        remove_stopwords: (bool) Remove stopwords
        apply_stemming: (bool) Apply stemming on the words on the text
    """
    #Remove emojis
    text = re.sub(":[a-zA-Z\-\_]*:","", emoji.demojize(text)) #:hear-no-evil_monkey:
    text = re.sub(":\w+:","", emoji.demojize(text))
    text = re.sub(":\w+\’\w+:","", emoji.demojize(text)) #:woman's_boot:

    #Remove mentions, usernames (@USER)
    text = re.sub("\s*@USER\s*", '', text)

    #Remove URL
    text = re.sub("\s*URL\s*", '', text)

    #And
    text = re.sub("&amp;", "and", text)
    text = re.sub("&lt;", "<", text)
    text = re.sub("&gt", ">", text)
    text = re.sub("&", "and", text)

    #Replace contractions and slang of word
    text = re.sub("i'm", "I'm", text)
    text = contractions.fix(text, slang=True)

    #Lowercase
    text = text.lower()

    #Remove Hashtags + Words
    text = re.sub("#\s*\w+\s*", '', text)

    #Remove repeating whitespaces
    text = re.sub("\s[2, ]"," ", text)

    #Remove non ascii characters
    text.encode("ascii", errors="ignore").decode()

    #Remove punctuations, numbers and special characters (remove emoticons)
    if remove_punt_number_special_chars:
        text = re.sub('[^a-zA-Z]', ' ', text)

    #Tokenize text
    tt = TweetTokenizer(preserve_case=False,
                    strip_handles=True,
                    reduce_len=True)

    text_tokens = tt.tokenize(text)

    #Remove stopwords
    if remove_stopwords:
        stopwords = set(STOPWORDS)
        text_tokens = [token for token in text_tokens if token not in stopwords]

    #Stemming
    if apply_stemming:
        text_stem = [stemmer.stem(token) for token in text_tokens]

    clean = " ".join(text_tokens)

    return clean

#BERT
def convert_examples_to_features(X, y, max_seq_length, tokenizer):

    """Loads a data file and returns examples (input_ids, input_mask, segment_ids, label_id).
    Args:
        data: Data
        max_seq_length: (int) Maximum length of the sequences
    """

    col_names = ["input_ids","input_mask","segment_ids","label_id"]
    features = pd.DataFrame(columns=col_names)

    df = pd.DataFrame({"text":X, "label":y})

    for index, example in df.iterrows():

        tokens_text = tokenizer.tokenize(example.text)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_text) > max_seq_length - 2:
            tokens_text = tokens_text[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_text + ["[SEP]"]
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
        os.makedirs(directory)

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
