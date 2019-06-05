import pandas as pd
import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report

import json
import logging
import os
import shutil

import torch

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
        #print(example)

        tokens_tweet = tokenizer.tokenize(example.tweet)
        #print(tokens_tweet)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_tweet) > max_seq_length - 2:
            tokens_tweet = tokens_tweet[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_tweet + ["[SEP]"]
        #print(tokens)
        #print("Length tokens:", len(tokens))
        segment_ids = [0] * len(tokens)
        #print(segment_ids)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #print(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        #print(example.label)
        label_id = int(example.label)
        #print(label_id)

        input_features = {'input_ids': input_ids, 'input_mask':input_mask, 'segment_ids':segment_ids, 'label_id':label_id}
        features.loc[len(features)] = input_features

    return features


#Logger
def set_logger(log_path):
    #Source: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py

    """
    Arguments:
        log_path: Path to log
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


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
