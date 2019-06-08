import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm, trange

from utils import accuracy_recall_precision_f1

import pandas as pd

def evaluate_model(model, optimizer, loss_fn, dataloader, device):
    """Evaluate model
    Args:
        model: Model either LSTM, LSTMAttention, CNN, MLP (torch.nn.Module)
        optimizer: Optimizer for parameters of the model (torch.optim)
        loss_fn: Loss function taht computs the loss for each batch based on the y_pred and y_target
        dataloader: Dataloader that generates batches of data and labels or in case of BERT input_ids, input_mask, segment_ids and label_ids
        device: Device run either on GPU or CPU
    """

    #Metrics
    epoch_loss = 0
    epoch_accuracy = 0
    epoch_recall = [0, 0]
    epoch_precision = [0, 0]
    epoch_f1 = [0, 0]

    model.eval()

    with torch.no_grad():

        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):

            #Step 0: Get batch
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            #Step 1: Compute the forward pass of the model (model output)
            y_pred = model(input_ids, segment_ids, input_mask, labels=None)

            #Step 2: Compute the loss
            loss = loss_fn(y_pred, y_target)
            loss_batch = loss.item()
            epoch_loss += loss_batch

            #Compute other metrics
            accuracy, recall, precision, f1 = accuracy_recall_precision_f1(y_pred, y_target)

            epoch_accuracy += accuracy
            epoch_recall += recall
            epoch_precision += precision
            epoch_f1 += f1

        #Evaluation results
        results = {
            'loss': epoch_loss / len(dataloader),
            'accuracy': float(epoch_accuracy / len(dataloader)),
            'recall': np.round(epoch_recall / len(dataloader), 2),
            'precision': np.round(epoch_precision / len(dataloader), 2),
            'f1': np.round(epoch_f1 / len(dataloader), 2)
        }

    return results
