import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
import logging

import click

import torch
from torch import nn
import torch.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from train import train_model
from evaluate import evaluate_model
from utils import accuracy_recall_precision_f1, set_logger, save_checkpoint, load_checkpoint
from data_loader import load_data, clean_data, get_dataloader, convert_examples_to_features

import warnings
warnings.filterwarnings('ignore')

#Sacred
#Sources
#https://github.com/gereleth/kaggle-telstra/blob/master/Automatic%20model%20tuning%20with%20Sacred%20and%20Hyperopt.ipynb
#https://github.com/maartjeth/sacred-_runample-pytorch

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment()
ex.observers.append(FileStorageObserver.create('runs'))

#Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def log_scalars(results, name_dataset):

    """Log scalars of the results for MongoDB and Omniboard
    Args:
        results: Results with the loss, accuracy, recall, precision and f1-score
        name_dataset: The name of the dataset so it can store the scalers by name
    """

    ex.log_scalar(name_dataset+'.loss', float(results['loss']))
    ex.log_scalar(name_dataset+'.accuracy', float(results['accuracy']))
    ex.log_scalar(name_dataset+'.recall.OFF', float(results['recall'][0]))
    ex.log_scalar(name_dataset+'.recall.NOT', float(results['recall'][1]))
    ex.log_scalar(name_dataset+'.precision.OFF', float(results['precision'][0]))
    ex.log_scalar(name_dataset+'.precision.NOT', float(results['precision'][1]))
    ex.log_scalar(name_dataset+'.f1.OFF', float(results['f1'][0]))
    ex.log_scalar(name_dataset+'.f1.NOT', float(results['f1'][1]))


@ex.capture
def train_and_evaluate(num_epochs, model, optimizer, loss_fn, dataloader, val_dataloader, scheduler, early_stopping_criteria, directory):

    """Train on training set and evaluate on evaluation set
    Args:
        num_epochs: Number of epochs to run the training and evaluation
        model: Model
        optimizer: Optimizer
        loss_fn: Loss function
        dataloader: Dataloader for the training set
        val_dataloader: Dataloader for the validation set
        scheduler: Scheduler
        directory: Directory path name to story the logging files

    Returns train and evaluation metrics with epoch, loss, accuracy, recall, precision and f1-score
    """

    metrics = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'recall', 'precision', 'f1'])
    val_metrics = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'recall', 'precision', 'f1'])

    best_val_loss = float("inf")

    early_stop_step = 0

    for epoch in trange(num_epochs, desc="Epoch"):

        ### TRAINING ###
        results = train_model(model, optimizer, loss_fn, dataloader, device)
        metrics.loc[len(metrics)] = {'epoch':epoch, 'loss':results['loss'], 'accuracy':results['accuracy'], 'recall':results['recall'], 'precision':results['precision'], 'f1':results['f1']}
        log_scalars(results, "Train")

        ### EVALUATION ###
        val_results = evaluate_model(model, optimizer, loss_fn, val_dataloader, device)
        val_metrics.loc[len(val_metrics)] = {'epoch':epoch, 'loss':val_results['loss'], 'accuracy':val_results['accuracy'], 'recall':val_results['recall'], 'precision':val_results['precision'], 'f1':val_results['f1']}
        log_scalars(results, "Validation")

        #Save best and latest state
        best_model = val_results['loss'] <= best_val_loss
        last_model = epoch == num_epochs-1

        if best_model:
            save_checkpoint({'epoch': epoch+1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                    directory=directory,
                                    checkpoint='best_model.pth.tar')

        if last_model:
            save_checkpoint({'epoch': epoch+1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                    directory=directory,
                                    checkpoint='last_model.pth.tar')

        #Early stopping
        if val_results['loss'] >= best_val_loss:
            early_stop_step += 1
        else:
            best_val_loss = val_results['loss']
            early_stop_step = 0

        stop_early = early_stop_step >= early_stopping_criteria

        if stop_early:
            print("Stopping early at epoch %i", epoch)
            break

        #Scheduler
        scheduler.step()

    return metrics, val_metrics

@ex.config
def congig():

    """Configuration"""

    num_labels = 2 #Number of labels (default=2)
    bs = 32 #Train batch size (default=32)
    val_bs = 32 #Validation batch size (default=32)
    test_bs = 32 #Test batch size (default=32)
    num_epochs = 1 #Number of epochs (default=1)
    max_seq_length = 40 #Maximum sequence length of the sentences (default=40)
    learning_rate = 3e-5 #Learning rate for the model (default=3e-5)
    warmup_proportion = 0.1 #Warmup proportion (default=0.1)
    early_stopping_criteria = 10 #Early stopping cri(default=10)

@ex.automain
def run(num_labels, bs, val_bs, test_bs, num_epochs, max_seq_length, learning_rate, early_stopping_criteria, warmup_proportion, _run):
    #Logger
    directory = f"runs/{_run._id}/"

    #Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    #Data
    print('Load datasets...')

    #Data
    data, val_data, test_data = load_data()
    df, val_df, test_df = clean_data(data), clean_data(val_data), clean_data(test_data)

    #Data _runamples
    examples = convert_examples_to_features(df.head(10), max_seq_length, tokenizer)
    val_examples = convert_examples_to_features(val_df.head(2), max_seq_length, tokenizer)
    test_examples = convert_examples_to_features(test_df, max_seq_length, tokenizer)

    #Dataloaders
    dataloader = get_dataloader(examples, bs)
    val_dataloader = get_dataloader(val_examples, val_bs)
    test_dataloader = get_dataloader(test_examples, test_bs)

    num_optimization_steps = int(len(df) / bs) * num_epochs

    #Model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    model = model.to(device)

    #Loss and optimizer
    optimizer = optim.Adam([
        {'params': model.parameters(), 'weight_decay': 0.1}], lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    #Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,15], gamma=0.1)

    #Training and evaluation
    print('Training and evaluation for {} epochs...'.format(num_epochs))
    train_metrics, val_metrics = train_and_evaluate(num_epochs, model, optimizer, loss_fn, dataloader, val_dataloader, scheduler, early_stopping_criteria, directory)
    train_metrics.to_csv(directory+"train_metrics.csv"), val_metrics.to_csv(directory+"val_metrics.csv")

    #Test
    print('Testing...')
    load_checkpoint(directory+"best_model.pth.tar", model)
    test_results = evaluate_model(model, optimizer, loss_fn, test_dataloader, device)
    log_scalars(test_results,"Test")

    test_results_df = pd.DataFrame(test_results, index=["NOT","OFF"])
    test_results_df = test_results_df.drop(columns=['loss','accuracy'])
    test_results_df.to_csv(directory+"test_metrics.csv")
