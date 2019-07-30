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
from sacred.utils import apply_backspaces_and_linefeeds

EXPERIMENT_NAME = 'experiment'
DATABASE_NAME = 'experiments'
URL_NAME = 'mongodb://localhost:27017/'

ex = Experiment()
#ex.observers.append(MongoObserver.create(url=URL_NAME, db_name=DATABASE_NAME))
ex.captured_out_filter = apply_backspaces_and_linefeeds

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
        #last_model = epoch == num_epochs-1

        if best_model:
            save_checkpoint({'epoch': epoch+1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                    directory=directory,
                                    checkpoint='best_model.pth.tar')

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
        print('Train recall: {} | Train precision: {} | Train f1: {}'.format(train_results['recall'], train_results['precision'], train_results['f1']))
        print('Valid recall: {} | Valid precision: {} | Valid f1 {}'.format(val_results['recall'], val_results['precision'], val_results['f1']))

        #Scheduler
        #scheduler.step()

    return train_metrics, val_metrics


@ex.config
def config():

    """Configuration"""

    output_dim = 2 #Number of labels (default=2)
    batch_size = 100 #Batch size (default=32)
    num_epochs = 10 #Number of epochs (default=100)
    max_seq_length = 45 #Maximum sequence length of the sentences (default=40)
    learning_rate = 3e-5 #Learning rate for the model (default=3e-5)
    warmup_proportion = 0.1 #Warmup proportion (default=0.1)
    early_stopping_criteria = 20 #Early stopping criteria (default=5)
    num_layers = 2 #Number of layers (default=2)
    hidden_dim = 128 #Hidden layers dimension (default=128)
    bidirectional = False #Left and right LSTM
    dropout = 0.5 #Dropout percentage
    filter_sizes = [2, 3, 4] #CNN
    embedding_file = 'data/GloVe/glove.twitter.27B.200d.txt' #Embedding file
    model_name = "MLP" #Model name: LSTM, BERT, MLP, CNN
    use_mongo = False
    subtask = "a" #Subtask name: a, b or c

    #ex.observers.append(MongoObserver.create(url=URL_NAME, db_name=DATABASE_NAME))
    if model_name == "MLP":
        ex.observers.append(FileStorageObserver.create('results-mlp'))
    elif model_name == "LSTM":
        ex.observers.append(FileStorageObserver.create('results-lstm'))
    elif model_name == "LSTMAttention":
        ex.observers.append(FileStorageObserver.create('results-lstmattention'))
    elif model_name == "CNN":
        ex.observers.append(FileStorageObserver.create('results-cnn'))
    elif "BERT" in model_name:
        ex.observers.append(FileStorageObserver.create('results-bert'))

@ex.automain
def main(output_dim,
        batch_size,
        num_epochs,
        max_seq_length,
        learning_rate,
        warmup_proportion,
        early_stopping_criteria,
        num_layers,
        hidden_dim,
        bidirectional,
        dropout,
        filter_sizes,
        embedding_file,
        model_name,
        use_mongo,
        subtask,
        _run):

    #Logger
    #directory_checkpoints = f"results/checkpoints/{_run._id}/"
    #directory = f"results/{_run._id}/"

    id_nummer = f'{_run._id}'
    print(id_nummer)

    if "BERT" in model_name:  #Default = False, if BERT model is used then use_bert is set to True
        use_bert = True
        directory = f"results-bert/{_run._id}/"
        directory_checkpoints =  f"results-bert/checkpoints/{_run._id}/"
    else:
        use_bert = False
        directory = f"results-"+model_name.lower()+"/"+id_nummer+"/"
        directory_checkpoints =  f"results-"+model_name.lower()+"/checkpoints"+"/"+id_nummer+"/"

    #Data
    if use_bert:
        train_dataloader, val_dataloader, test_dataloader = get_data_bert(int(max_seq_length), batch_size, subtask)
    else:
        embedding_dim, vocab_size, embedding_matrix, train_dataloader, val_dataloader, test_dataloader = get_data(int(max_seq_length), embedding_file=embedding_file, batch_size=batch_size, subtask=subtask)

    #Model
    if model_name=="MLP":
        model = models.MLP(embedding_matrix, embedding_dim, vocab_size, int(hidden_dim), dropout, output_dim)
    if model_name=="MLP_Features":
        model = models.MLP_Features(embedding_matrix, embedding_dim, vocab_size, int(hidden_dim), 14, dropout, output_dim)
        #print(model)
    elif model_name=="CNN":
        model = models.CNN(embedding_matrix, embedding_dim, vocab_size, dropout, filter_sizes, output_dim)
        #print(model)
    elif model_name=="LSTM":
        model = models.LSTM(embedding_matrix, embedding_dim, vocab_size, int(hidden_dim), dropout, int(num_layers), bidirectional, output_dim)
        #print(model)
    elif model_name=="LSTMAttention":
        model = models.LSTMAttention(embedding_matrix, embedding_dim, vocab_size, int(hidden_dim), dropout, int(num_layers), bidirectional, output_dim)
        #print(model)
    elif model_name=="BERTFreeze":
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", output_dim)
        for param in model.bert.parameters():
            param.requires_grad = False
            print(param)
            print(param.requires_grad)
        print(model)
    elif model_name=="BERT":
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", output_dim)
        #print(model)
    elif model_name=="BERTLinear":
        model = models.BertLinear(hidden_dim, dropout, output_dim)
        #print(model)
    elif model_name=="BERTLinearFreeze":
        model = models.BertLinearFreeze(hidden_dim, dropout, output_dim)
        #print(model)
    elif model_name=="BERTLinearFreezeEmbeddings":
        model = models.BertLinearFreezeEmbeddings(hidden_dim, dropout, output_dim)
        #print(model)
    elif model_name=="BERTLSTM":
        model = models.BertLSTM(hidden_dim, dropout, bidirectional, output_dim)
        #print(model)
    elif model_name=="BERTNonLinear":
        model = models.BertNonLinear(dropout, output_dim)
        #print(model)
    elif model_name=="BERTNorm":
        model = models.BertNorm(dropout, output_dim)
        #print(model)
    elif model_name=="BERTPooling":
        model = models.BertPooling(dropout, output_dim)
    elif model_name=="BERTExtractEmbeddings":
        model = models.BertExtractEmbeddings(dropout, output_dim)
    else:
        Exception("Model name not recoginized")
        return
    model = model.to(device)

    #Loss and optimizer
    #optimizer = optim.Adam([{'params': model.parameters(), 'weight_decay': 0.1}], lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = F.cross_entropy

    #Scheduler
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 50], gamma=0.1)

    #Training and evaluation
    print('Training and evaluation for {} epochs...'.format(num_epochs))
    train_metrics, val_metrics = train_and_evaluate(num_epochs, model, optimizer, loss_fn, train_dataloader, val_dataloader, early_stopping_criteria, directory_checkpoints, use_bert, use_mongo)
    train_metrics.to_csv(directory+"train_metrics.csv"), val_metrics.to_csv(directory+"val_metrics.csv")

    #Test
    print('Testing...')
    load_checkpoint(directory_checkpoints+"best_model.pth.tar", model)

    #Add artifacts
    #ex.add_artifact(directory+"best_model.pth.tar")

    test_metrics = evaluate_model(model, optimizer, loss_fn, test_dataloader, device, use_bert)
    if use_mongo: log_scalars(test_metrics,"Test")

    #test_metrics_df = pd.DataFrame(test_metrics)
    print(test_metrics)
    #test_metrics_df.to_csv(directory+"test_metrics.csv")

    results = {
        'id': id_nummer,
        #'loss': np.round(np.mean(val_metrics['loss']), 4),
        'loss': 1-test_metrics['accuracy'],
        'accuracy': test_metrics['accuracy'],
        'recall': test_metrics['recall'],
        'precision': test_metrics['precision'],
        'f1': test_metrics['f1'],
        'learning_rate': learning_rate,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'max_seq_length': max_seq_length,
        'status': 'ok'
    }

    return results
