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
from torch.nn import functional as F
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
from utils import accuracy_recall_precision_f1, save_checkpoint, load_checkpoint
from data_loader import get_data, get_data_bert
import models

import warnings
warnings.filterwarnings('ignore')

#Sacred
#Sources
#https://github.com/gereleth/kaggle-telstra/blob/master/Automatic%20model%20tuning%20with%20Sacred%20and%20Hyperopt.ipynb
#https://github.com/maartjeth/sacred-example-pytorch

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sacred.observers import SlackObserver

EXPERIMENT_NAME = 'experiment'
DATABASE_NAME = 'experiments'
URL_NAME = 'mongodb://localhost:27017/'

ex = Experiment()
ex.observers.append(FileStorageObserver.create('results'))

#Send a message to slack if the run is succesfull or if it failed
slack_obs = SlackObserver.from_config('slack.json')
ex.observers.append(slack_obs)

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
def train_and_evaluate(num_epochs, model, optimizer, loss_fn, train_dataloader, val_dataloader, early_stopping_criteria, directory, use_bert, use_mongo):

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

    train_metrics = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'recall', 'precision', 'f1'])
    val_metrics = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'recall', 'precision', 'f1'])

    best_val_loss = float("inf")

    early_stop_step = 0

    for epoch in trange(num_epochs, desc="Epoch"):

        ### TRAINING ###
        train_results = train_model(model, optimizer, loss_fn, train_dataloader, device, use_bert)
        train_metrics.loc[len(train_metrics)] = {'epoch':epoch, 'loss':train_results['loss'], 'accuracy':train_results['accuracy'], 'recall':train_results['recall'], 'precision':train_results['precision'], 'f1':train_results['f1']}
        if use_mongo: log_scalars(train_results, "Train")

        ### EVALUATION ###
        val_results = evaluate_model(model, optimizer, loss_fn, val_dataloader, device, use_bert)
        val_metrics.loc[len(val_metrics)] = {'epoch':epoch, 'loss':val_results['loss'], 'accuracy':val_results['accuracy'], 'recall':val_results['recall'], 'precision':val_results['precision'], 'f1':val_results['f1']}
        if use_mongo: log_scalars(val_results, "Validation")

        #Save best and latest state
        best_model = val_results['loss'] < best_val_loss
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
            print("Early stop step:", early_stop_step)
        else:
            best_val_loss = val_results['loss']
            early_stop_step = 0

        stop_early = early_stop_step >= early_stopping_criteria

        if stop_early:
            print("Stopping early at epoch {}".format(epoch))
            return train_metrics, val_metrics

        print('\n')
        print('Train Loss: {} | Train Acc: {}'.format(train_results['loss'], train_results['accuracy']))
        print('Valid Loss: {} | Valid Acc: {}'.format(val_results['loss'], val_results['accuracy']))
        print('Train recall: {} | Train precision: {}'.format(train_results['recall'], train_results['precision']))
        print('Valid recall: {} | Valid precision: {}'.format(val_results['recall'], val_results['precision']))

        #Scheduler
        #scheduler.step()

    return train_metrics, val_metrics


@ex.config
def config():

    """Configuration"""

    output_dim = 2 #Number of labels (default=2)
    train_bs = 32.0 #Train batch size (default=32)
    val_bs = 32.0 #Validation batch size (default=32)
    test_bs = 32.0  #Test batch size (default=32)
    num_epochs = 100 #Number of epochs (default=100)
    max_seq_length = 45 #Maximum sequence length of the sentences (default=40)
    learning_rate = 3e-3 #Learning rate for the model (default=3e-5)
    warmup_proportion = 0.1 #Warmup proportion (default=0.1)
    #early_stopping_criteria = 50 #Early stopping criteria (default=5)
    embedding_dim = 100 #Embedding dimension (default=100)
    num_layers = 2 #Number of layers (default=2)
    hidden_dim = 128 #Hidden layers dimension (default=128)
    bidirectional = True #Left and right LSTM
    dropout = 0.1 #Dropout percentage
    filter_sizes = [2, 3, 4] #CNN
    embedding_file = 'data/GloVe/glove.twitter.27B.200d.txt' #Embedding file
    model_name = "MLP" #Model name: LSTM, BERT, MLP, CNN
    use_mongo = False

@ex.automain
def run(output_dim,
        train_bs,
        val_bs,
        test_bs,
        num_epochs,
        max_seq_length,
        learning_rate,
        warmup_proportion,
        #early_stopping_criteria,
        embedding_dim,
        num_layers,
        hidden_dim,
        bidirectional,
        dropout,
        filter_sizes,
        embedding_file,
        model_name,
        use_mongo,
        _run):

    #Mongo
    if use_mongo: ex.observers.append(MongoObserver.create(url=URL_NAME, db_name=DATABASE_NAME))

    #Logger
    directory_checkpoint = f"results/checkpoints/{_run._id}/"
    directory = f"results/{_run._id}/"

    #Batch sizes
    batch_sizes = [int(train_bs), int(val_bs), int(test_bs)]
    batch_size = int(train_bs)

    if "Bert" in model_name:  #Default = False, if BERT model is used then use_bert is set to True
        use_bert = True
    else:
        use_bert = False

    #Data
    if use_bert:
        train_dataloader, val_dataloader, test_dataloader = get_data_bert(max_seq_length, batch_sizes)
    else:
        embedding_dim, vocab_size, embedding_matrix, train_dataloader, val_dataloader, test_dataloader = get_data(max_seq_length, embedding_file=embedding_file, batch_size=batch_size)

    #Model
    if model_name=="MLP":
        model = models.MLP(vocab_size, embedding_dim, embedding_matrix, hidden_dim, dropout, output_dim)
        print(model)
    elif model_name=="CNN":
        model = models.CNN(vocab_size, embedding_dim, embedding_matrix, output_dim, filter_sizes, embedding_dim, dropout)
        print(model)
    elif model_name=="LSTM":
        model = models.LSTM(embedding_matrix, num_layers, hidden_dim, bidirectional, vocab_size, embedding_dim, dropout, output_dim)
        print(model)
    elif model_name=="LSTMAttention":
        model = models.LSTMAttention(embedding_matrix, hidden_dim, vocab_size, embedding_dim, output_dim, batch_size)
        print(model)
    elif model_name=="BertLinear":
        #model = BertForSequenceClassification.from_pretrained("bert-base-uncased", output_dim)
        model = models.BertLinear(dropout, output_dim)
        print(model)
    elif model_name=="BertLSTM":
        model = models.BertLSTM(dropout, output_dim)
        print(model)

    model = model.to(device)

    #Loss and optimizer
    #optimizer = optim.Adam([{'params': model.parameters(), 'weight_decay': 0.1}], lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = F.cross_entropy

    #Scheduler
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,15], gamma=0.1)

    #Training and evaluation
    print('Training and evaluation for {} epochs...'.format(num_epochs))
    train_metrics, val_metrics = train_and_evaluate(num_epochs, model, optimizer, loss_fn, train_dataloader, val_dataloader, int(num_epochs/2), directory_checkpoint, use_bert, use_mongo)
    #print(train_metrics); print(val_metrics)
    train_metrics.to_csv(directory+"train_metrics.csv"), val_metrics.to_csv(directory+"val_metrics.csv")

    #Test
    print('Testing...')
    load_checkpoint(directory_checkpoint+"best_model.pth.tar", model)
    test_metrics = evaluate_model(model, optimizer, loss_fn, test_dataloader, device, use_bert)
    if use_mongo: log_scalars(test_metrics,"Test")

    test_metrics_df = pd.DataFrame(test_metrics, index=["NOT","OFF"])
    test_metrics_df = test_metrics_df.drop(columns=['loss','accuracy'])
    print(test_metrics)
    test_metrics_df.to_csv(directory+"test_metrics.csv")

    id_nummer = f'{_run._id}'
    print("ID:", id_nummer)

    results = {
        # 'train_loss': np.mean(train_metrics['loss'])/len(train_metrics), 2),
        # 'train_accuracy': np.mean(train_metrics['accuracy'])
        #'train_recall': train_metrics['recall'] / len(train_metrics),
        #'train_precision': train_metrics['precision'] / len(train_metrics),
        #'train_f1': train_metrics['f1'] / len(train_metrics),
        'id': id_nummer,
        'loss': np.mean(val_metrics['loss']),
        # 'val_accuracy': np.mean(val_metrics['accuracy']),
        #'val_recall': val_metrics['recall'] / len(val_metrics),
        #'val_precision': val_metrics['precision'] / len(val_metrics),
        #'val_f1': val_metrics['f1'] / len(val_metrics)
        'accuracy': test_metrics['accuracy'],
        'recall': test_metrics['recall'],
        'precision': test_metrics['precision'],
        'f1': test_metrics['f1'],
        'learning_rate': learning_rate,
        'status': 'ok'
    }

    #return np.round(sum(train_metrics['loss'])/len(train_metrics), 2), 'status': 'ok'
    return results
