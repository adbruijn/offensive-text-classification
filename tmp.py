from data_loader import get_data
import models

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
from train import train_model
from evaluate import evaluate_model

#Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

batch_sizes = [100, 100, 100]
embedding_dim = 100
hidden_dim = 100
output_dim = 2
learning_rate = 0.0001
num_epochs = 2

#vocab_size, embedding_matrix, train_dataloader, val_dataloader, test_dataloader = get_data2(70, batch_sizes, embedding_file='data/GloVe/glove.6B.100d.txt')
vocab_size, embedding_matrix, train_iter, valid_iter, test_iter = get_data(max_seq_len=70, embedding_file='data/GloVe/glove.6B.100d.txt', batch_size=100)

model = models.MLP(vocab_size, embedding_dim, embedding_matrix, hidden_dim, output_dim)
print(model)

#Loss and optimizer
optimizer = optim.Adam(model.parameters())
loss_fn = F.cross_entropy

##########
train_metrics = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'recall', 'precision', 'f1'])
val_metrics = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'recall', 'precision', 'f1'])

best_val_loss = float("inf")

early_stop_step = 0

for epoch in trange(num_epochs, desc="Epoch"):

    ### TRAINING ###
    print("hoi")
    #train_results = train_model(model, optimizer, loss_fn, train_iter, device, use_bert=False)
    #train_metrics.loc[len(train_metrics)] = {'epoch':epoch, 'loss':train_results['loss'], 'accuracy':train_results['accuracy'], 'recall':train_results['recall'], 'precision':train_results['precision'], 'f1':train_results['f1']}
    #log_scalars(train_results, "Train")

    # ### EVALUATION ###
    val_results = evaluate_model(model, optimizer, loss_fn, valid_iter, device, use_bert=False)
    # val_metrics.loc[len(val_metrics)] = {'epoch':epoch, 'loss':val_results['loss'], 'accuracy':val_results['accuracy'], 'recall':val_results['recall'], 'precision':val_results['precision'], 'f1':val_results['f1']}
    # log_scalars(val_results, "Validation")
    #
    # #Save best and latest state
    # best_model = val_results['loss'] <= best_val_loss
    # last_model = epoch == num_epochs-1
    #
    # if best_model:
    #     save_checkpoint({'epoch': epoch+1,
    #                            'state_dict': model.state_dict(),
    #                            'optim_dict': optimizer.state_dict()},
    #                             directory=directory,
    #                             checkpoint='best_model.pth.tar')
    #
    # if last_model:
    #     save_checkpoint({'epoch': epoch+1,
    #                            'state_dict': model.state_dict(),
    #                            'optim_dict': optimizer.state_dict()},
    #                             directory=directory,
    #                             checkpoint='last_model.pth.tar')
    #
    # #Early stopping
    # if val_results['loss'] >= best_val_loss:
    #     early_stop_step += 1
    #     print("Early stop step:", early_stop_step)
    # else:
    #     best_val_loss = val_results['loss']
    #     early_stop_step = 0
    #
    # stop_early = early_stop_step >= early_stopping_criteria
    #
    # if stop_early:
    #     print("Stopping early at epoch {}".format(epoch))
    #     return train_metrics, val_metrics
    #
    # #Scheduler
    # scheduler.step()
