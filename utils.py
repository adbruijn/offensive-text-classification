import pandas as pd
import torch
import numpy as np

import json
import logging
import os
import shutil

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report, confusion_matrix

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
import string
from textblob import TextBlob

stopwords = stopwords.words('english')

from gensim.models import KeyedVectors
import operator
import wordsegment
import re
import contractions
from nltk.tokenize import TweetTokenizer
from wordsegment import load, segment
from ekphrasis.classes.tokenizer import SocialTokenizer
import emoji_extra

tt = TweetTokenizer(preserve_case=False,
                    strip_handles=True,
                    reduce_len=True)

social_tokenizer = SocialTokenizer(lowercase=True).tokenize

load()

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

    # features['polarity'] = features['text'].apply(lambda x: np.round(TextBlob(x).sentiment.polarity, 2))
    # features['subjectivity'] = features['text'].apply(lambda x: np.round(TextBlob(x).sentiment.subjectivity, 2))

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

    return cm

#Hashtag
def segment_hashtag_word2vec(match):
    """
    Takes a re 'match object' and segments the hashtag
    """

    hashtag_body = match.group(1)
    result = " hashtag {} ".format(" ".join(segment(hashtag_body)))
    return result

def segment_hashtag_glove(match):
    """
    Takes a re 'match object' and segments the hashtag
    """

    hashtag_body = match.group(1)
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(" ".join(segment(hashtag_body)))
    else:
        result = "<hashtag> {} ".format(" ".join(segment(hashtag_body)))

    return result

def segment_hashtag_fasttext(match):
    """
    Takes a re 'match object' and segments the hashtag
    """

    hashtag_body = match.group(1)
    if hashtag_body.isupper():
        result = " hashtag {} allcaps ".format(" ".join(segment(hashtag_body)))
    else:
        result = " hashtag {} ".format(" ".join(segment(hashtag_body)))

    return result

def clean_glove(text):

    #Remove mentions, usernames (@USER)
    text = re.sub("\s*@USER\s*", '<user>', text)

    #Remove URL
    text = re.sub("\s*URL\s*", '<url>', text)

    #Numbers
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", text)

    #Emoticons
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    text = re.sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>", text)
    text = re.sub(r"{}{}p+".format(eyes, nose), "<lolface>", text)
    text = re.sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>", text)
    text = re.sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>", text)

    #Emojis
    text = emoji_extra.demojize(text)

    #Remove skin tones
    text = re.sub(" medium-dark skin tone", "", text)
    text = re.sub(" medium-light skin tone", "", text)
    text = re.sub(" medium skin tone", "", text)
    text = re.sub(" dark skin tone", "", text)
    text = re.sub(" light skin tone", "", text)

    #Hashtag
    #Source: https://pypi.org/project/wordsegment/
    text = re.sub(r"#(\w+)", segment_hashtag_glove, text)

    #Repeat
    # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
    text = re.sub(r"([!?.]){2,}", r"\1 <repeat>", text)

    #Elong
    # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
    text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>", text)

    #Lowercase
    text = re.sub(r"([A-Z]){2,}", text.lower() + " <allcaps>", text)
    text = text.lower()

    #Replace contractions and slang of word
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")

    text = re.sub("i'm", "i am", text)
    text = re.sub("i've", "i have", text)
    text = contractions.fix(text, slang=True)

    #text = " ".join(tt.tokenize(text))
    text = " ".join(social_tokenizer(text))

    return text

def clean_word2vec(text):

    #Remove mentions, usernames (@USER)
    text = re.sub("\s*@USER\s*", ' user ', text)

    #Remove URL
    text = re.sub("\s*URL\s*", ' url ', text)

    #Numbers
    #text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", text)
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)

    #Emojis
    text = emoji_extra.demojize(text)

    #Remove skin tones
    text = re.sub(" medium-dark skin tone", "", text)
    text = re.sub(" medium-light skin tone", "", text)
    text = re.sub(" medium skin tone", "", text)
    text = re.sub(" dark skin tone", "", text)
    text = re.sub(" light skin tone", "", text)

    #Hashtag
    #Source: https://pypi.org/project/wordsegment/
    text = re.sub(r"#(\w+)", segment_hashtag_word2vec, text)

    #Punctuation
    text = re.sub("&amp;", "and", text)
    text = re.sub("&lt;", "<", text)
    text = re.sub("&gt", ">", text)
    text = re.sub("&", "and", text)

    #Lowercase
    text = text.lower()

    #Replace contractions and slang of word
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")

    text = re.sub("i'm", "i am", text)
    #text = re.sub("i’m", "i am", text)
    text = re.sub("i've", "i have", text)
    #text = re.sub("i’ve", "i have", text)
    text = contractions.fix(text, slang=True)

    #Punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    #text = " ".join(tt.tokenize(text))
    stopwords = ['to','and','a','of']
    tokenized_text = social_tokenizer(text)
    filtered_stopwords = [w for w in tokenized_text if w not in stopwords]

    text = " ".join(filtered_stopwords)

    return text

def clean_fasttext(text):

    #Remove mentions, usernames (@USER)
    text = re.sub("\s*@USER\s*", ' user ', text)

    #Remove URL
    text = re.sub("\s*URL\s*", ' url ', text)

    #Emojis
    text = emoji_extra.demojize(text)

    #Remove skin tones
    text = re.sub(" medium-dark skin tone", "", text)
    text = re.sub(" medium-light skin tone", "", text)
    text = re.sub(" medium skin tone", "", text)
    text = re.sub(" dark skin tone", "", text)
    text = re.sub(" light skin tone", "", text)

    #Hashtag
    #Source: https://pypi.org/project/wordsegment/
    text = re.sub(r"#(\w+)", segment_hashtag_fasttext, text)

    #Lowercase
    text = re.sub(r"([A-Z]){2,}", text.lower() + " allcaps ", text)
    text = text.lower()

    #Replace contractions and slang of word
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")

    text = re.sub("i'm", "i am", text)
    text = re.sub("i've", "i have", text)
    text = contractions.fix(text, slang=True)

    #text = " ".join(tt.tokenize(text))
    text = " ".join(social_tokenizer(text))

    return text

def clean_text(text, remove_hashtags=False, remove_emojis=False,remove_punt_number_special_chars=False,remove_stopwords=False, apply_stemming=False):
    """Clean text
    Args:
        text: (str) Text
        remove_punt_number_special_chars: (bool) Remove punctuations, numbers and special characters
        remove_stopwords: (bool) Remove stopwords
        apply_stemming: (bool) Apply stemming on the words on the text
    """
    #Remove emojis
    if(remove_emojis):
        text = re.sub(":[a-zA-Z\-\_]*:","", emoji.demojize(text)) #:hear-no-evil_monkey:
        text = re.sub(":\w+:","", emoji.demojize(text))
        text = re.sub(":\w+\’\w+:","", emoji.demojize(text)) #:woman's_boot:
    else:
        text = emoji_extra.demojize(text)

        #Remove skin tones
        text = re.sub(" medium-dark skin tone", "", text)
        text = re.sub(" medium-light skin tone", "", text)
        text = re.sub(" medium skin tone", "", text)
        text = re.sub(" dark skin tone", "", text)
        text = re.sub(" light skin tone", "", text)

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
    if(remove_hashtags):
        text = re.sub("#\s*\w+\s*", '', text)
    else:
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
