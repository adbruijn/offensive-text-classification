import numpy as np
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

import pandas as pd

import json
import logging
import os
import shutil

from train import train_model
from evaluate import evaluate_model
from utils import accuracy_recall_precision_f1, set_logger, save_checkpoint, load_checkpoint
from data_loader import load_data, clean_data, get_dataloader, convert_examples_to_features

def train_and_evaluate(num_epochs, model, optimizer, loss_fn, train_dataloader, val_dataloader, scheduler, device, directory):
    train_metrics = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'recall', 'precision', 'f1'])
    val_metrics = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'recall', 'precision', 'f1'])

    best_val_loss = 100

    early_stop_step = 0

    for epoch in trange(num_epochs, desc="Epoch"):

        ### TRAINING ###
        train_results = train_model(model, optimizer, loss_fn, train_dataloader, device)
        train_metrics.loc[len(train_metrics)] = {'epoch':epoch, 'loss':train_results['loss'], 'accuracy':train_results['accuracy'], 'recall':train_results['recall'], 'precision':train_results['precision'], 'f1':train_results['f1']}

        #Logging
        train_results_string = " ; ".join("{}: {}".format(k, v) for k, v in train_results.items())
        logging.info("\nTrain metrics for epoch %i: %s", epoch, train_results_string)

        ### EVALUATION ###
        val_results = evaluate_model(model, optimizer, loss_fn, val_dataloader, device)
        val_metrics.loc[len(val_metrics)] = {'epoch':epoch, 'loss':val_results['loss'], 'accuracy':val_results['accuracy'], 'recall':val_results['recall'], 'precision':val_results['precision'], 'f1':val_results['f1']}

        #Logging
        val_results_string = " ; ".join("{}: {}".format(k, v) for k, v in val_results.items())
        logging.info("\nEvaluation metrics for epoch %i: %s", epoch, val_results_string)

        #Save best and latest state
        best_model = val_results['loss'] <= best_val_loss
        last_model = epoch == num_epochs

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
            logging.info("Eearly %s", early_stop_step)
        else:
            best_val_loss = val_results['loss']
            early_stop_step = 0

        stop_early = early_stop_step >= early_stopping_criteria

        if stop_early:
            logging.info("Stop early at epoch %i", epoch)
            break

        #Scheduler
        scheduler.step()

    return train_metrics, val_metrics

#Parameters: Using command line interface Click (https://click.palletsprojects.com/en/7.x/)
@click.command()
@click.option('--num_labels', default=2, show_default=True, help='Number of labels')
@click.option('--train_bs', default=32, show_default=True, help='Train batch size')
@click.option('--val_bs', default=32, show_default=True, help='Validation batch size')
@click.option('--test_bs', default=32, show_default=True, help='Test batch size')
@click.option('--num_epochs', default=100, show_default=True, help='Number of epochs')
@click.option('--max_seq_length', default=40, show_default=True, help='Maximum sequence length of the sentences')
@click.option('--learning_rate', default=3e-5, show_default=True, help='Learning rate for the model')
@click.option('--warmup_proportion', default=0.1, show_default=True, help='Warmup proportion')
@click.option('--directory', default="data/model/", show_default=True, help='Directory to save the results')

def run(num_labels, train_bs, val_bs, test_bs, num_epochs, max_seq_length, learning_rate, warmup_proportion, directory):
    #Logger
    set_logger(os.path.join(directory, 'train.log'))

    #Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    #Data
    logging.info('Load datasets')

    #Data
    train_data, val_data, test_data = load_data()
    train_df, val_df, test_df = clean_data(train_data), clean_data(val_data), clean_data(test_data)

    #Data examples
    train_examples = convert_examples_to_features(train_df.head(100), max_seq_length, tokenizer)
    val_examples = convert_examples_to_features(val_df.head(20), max_seq_length, tokenizer)
    test_examples = convert_examples_to_features(test_df, max_seq_length, tokenizer)

    #Dataloaders
    train_dataloader = get_dataloader(train_examples, train_bs)
    val_dataloader = get_dataloader(val_examples, val_bs)
    test_dataloader = get_dataloader(test_examples, test_bs)
    logging.info('- Done')

    num_train_optimization_steps = int(len(train_df) / train_bs) * num_epochs

    #Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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
    import time
    start_time = time.time()
    logging.info('Start training and evaluation for %s epochs', num_epochs)
    train_metrics, val_metrics = train_and_evaluate(num_epochs, model, optimizer, loss_fn, train_dataloader, val_dataloader, scheduler, device, directory)
    train_metrics.to_csv(directory+"train_metrics.csv")
    val_metrics.to_csv(directory+"val_metrics.csv")
    logging.info('- Done')
    print("--- %s seconds ---" % (time.time() - start_time))

    #Test
    logging.info('Start testing')
    load_checkpoint(directory+"best_model.pth.tar", model)
    test_results = evaluate_model(model, optimizer, loss_fn, test_dataloader, device)
    print("Test Results: ", test_results)
    test_results.to_csv(directory+"test_metrics.csv")
    logging.info('- Done')

if __name__ == '__main__':
    run()
