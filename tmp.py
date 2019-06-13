# import time
#
# start_time = time.time()
#
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.nn import functional as F
# import torch.optim as optim
# import numpy as np
#
# from tqdm import tqdm, tqdm_notebook, tnrange
# tqdm.pandas(desc='Progress')
#
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.nn import functional as F
#
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
#
# from tqdm import tqdm, trange
# import pandas as pd
# import time
#
# def epoch_time(start_time, end_time):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs
#
# from loader import load_data
#
# vocab_size, embedding_matrix, train_iter, valid_iter, test_iter = load_data(max_num_words=1000000, embedding_dim=100, max_seq_len=70, embedding_file='data/GloVe/glove.6B.100d.txt', batch_size=100)
#
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.nn import functional as F
# import numpy as np
# import torch.optim as optim
#
# NUM_LAYERS = 2
# HIDDEN_DIM = 100
# BIDIRECTIONAL = True
# VOCAB_SIZE = vocab_size
# EMBEDDING_DIM = 100
# DROPOUT = 0.1
# OUTPUT_DIM = 2
# BATCH_SIZE = 100
#
# import models
# model = models.LSTM(embedding_matrix, NUM_LAYERS, HIDDEN_DIM, BIDIRECTIONAL, VOCAB_SIZE, EMBEDDING_DIM, DROPOUT, OUTPUT_DIM)
# optimizer = optim.Adam(model.parameters())
# criterion = F.cross_entropy
#
# def train(model, iterator, optimizer, criterion):
#
#     epoch_loss = 0
#     epoch_acc = 0
#     epoch_precision = 0
#     epoch_recall = 0
#     epoch_f1 = 0
#
#     model.train()
#
#     for data, target in tqdm(iterator, desc="Iteration"):
#
#         optimizer.zero_grad()
#         target = torch.autograd.Variable(target).long()
#
#         #1. Forward propagation
#         prediction = model(data)
#
#         ## 2. Loss calculation
#         loss = criterion(prediction, target)
#
#         #3. Backward propagation
#         loss.backward()
#
#         #4. Weight optimization
#         optimizer.step()
#
#         preds = torch.max(prediction, 1)[1].view(target.size()).data
#
#         num_corrects = (preds == target.data).sum()
#         acc = num_corrects/len(data)
#
#         precision = precision_score(target, preds, average=None)
#         recall = recall_score(target, preds, average=None)
#         f1 = f1_score(target, preds, average=None)
#
#         micro = precision_recall_fscore_support(target, preds, average='micro')
#         macro = precision_recall_fscore_support(target, preds, average='macro')
#         weighted = precision_recall_fscore_support(target, preds, average='weighted')
#
#         epoch_loss += loss.item()
#         epoch_acc += acc
#         epoch_recall += recall
#         epoch_f1 += f1
#         epoch_precision += precision
#
#     return epoch_loss / len(iterator), acc, epoch_recall / len(iterator), epoch_precision / len(iterator), epoch_f1 / len(iterator)
# def evaluate(model, iterator, criterion):
#
#     epoch_loss = 0
#     epoch_acc = 0
#     epoch_precision = 0
#     epoch_recall = 0
#     epoch_f1 = 0
#
#     model.eval()
#
#     with torch.no_grad():
#
#         for data, target in tqdm(iterator, desc="Iteration"):
#
#             target = torch.autograd.Variable(target).long()
#
#             #1. Forward propagation
#             prediction = model(data)
#
#             ## 2. Loss calculation
#             loss = criterion(prediction, target)
#
#             preds = torch.max(prediction, 1)[1].view(target.size()).data
#
#             num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
#             acc = num_corrects/len(data)
#
#             precision = precision_score(target, preds, average=None)
#             recall = recall_score(target, preds, average=None)
#             f1 = f1_score(target, preds, average=None)
#
#             epoch_loss += loss.item()
#             epoch_acc += acc
#             epoch_recall += recall
#             epoch_f1 += f1
#             epoch_precision += precision
#
#     return epoch_loss / len(iterator), acc, epoch_recall / len(iterator), epoch_precision / len(iterator), epoch_f1 / len(iterator)
#
# #TRAIN AND EVALUATE
# # N_EPOCHS = 10
# #
# # best_valid_loss = float('inf')
# #
# # for epoch in trange(N_EPOCHS, desc="Epoch"):
# #
# #     start_time = time.time()
# #
# #     train_loss, train_acc, train_recall, train_precision, train_f1 = train(model, train_iter, optimizer, criterion)
# #     valid_loss, valid_acc, valid_recall, valid_precision, valid_f1 = evaluate(model, valid_iter, criterion)
# #
# #     end_time = time.time()
# #
# #     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
# #
# #     if valid_loss < best_valid_loss:
# #         best_valid_loss = valid_loss
# #         torch.save(model.state_dict(), 'lstm-model.pt')
# #
# #     print('Epoch:{}'.format(epoch))
# #     print('Train Loss: {} | Train Acc: {}'.format(train_loss, train_acc))
# #     print('Val. Loss: {} | Val. Acc: {}'.format(valid_loss, valid_acc))
# #     print('Train recall: {} | Train precision: {}'.format(train_recall, train_precision))
# #     print('Val. recall: {} | Val. precision: {}'.format(valid_recall, valid_precision))
# #
# # model.load_state_dict(torch.load('lstm-model.pt'))
# #
# # test_loss, test_acc, test_recall, test_precision, test_f1 = evaluate(model, test_iter, criterion)
# #
# # print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
# # print(test_precision, test_recall, test_f1)
# from train import train_model
# from evaluate import evaluate_model
# def train_and_evaluate(num_epochs, model, optimizer, loss_fn, train_dataloader, val_dataloader, early_stopping_criteria, directory, use_bert):
#
#     """Train on training set and evaluate on evaluation set
#     Args:
#         num_epochs: Number of epochs to run the training and evaluation
#         model: Model
#         optimizer: Optimizer
#         loss_fn: Loss function
#         dataloader: Dataloader for the training set
#         val_dataloader: Dataloader for the validation set
#         scheduler: Scheduler
#         directory: Directory path name to story the logging files
#
#     Returns train and evaluation metrics with epoch, loss, accuracy, recall, precision and f1-score
#     """
#
#     train_metrics = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'recall', 'precision', 'f1'])
#     val_metrics = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'recall', 'precision', 'f1'])
#
#     best_val_loss = float("inf")
#
#     early_stop_step = 0
#
#     for epoch in trange(num_epochs, desc="Epoch"):
#
#         ### TRAINING ###
#         train_results = train_model(model, optimizer, loss_fn, train_dataloader, device, use_bert)
#         train_metrics.loc[len(train_metrics)] = {'epoch':epoch, 'loss':train_results['loss'], 'accuracy':train_results['accuracy'], 'recall':train_results['recall'], 'precision':train_results['precision'], 'f1':train_results['f1']}
#
#         ### EVALUATION ###
#         val_results = evaluate_model(model, optimizer, loss_fn, val_dataloader, device, use_bert)
#         val_metrics.loc[len(val_metrics)] = {'epoch':epoch, 'loss':val_results['loss'], 'accuracy':val_results['accuracy'], 'recall':val_results['recall'], 'precision':val_results['precision'], 'f1':val_results['f1']}
#
#         #Save best and latest state
#         best_model = val_results['loss'] <= best_val_loss
#         last_model = epoch == num_epochs-1
#
#         #Early stopping
#         if val_results['loss'] >= best_val_loss:
#             early_stop_step += 1
#             print("Early stop step:", early_stop_step)
#         else:
#             best_val_loss = val_results['loss']
#             early_stop_step = 0
#
#         stop_early = early_stop_step >= early_stopping_criteria
#
#         if stop_early:
#             print("Stopping early at epoch {}".format(epoch))
#             return train_metrics, val_metrics
#
#         print('Epoch:{}'.format(epoch))
#         print('Train Loss: {} | Train Acc: {}'.format(train_results['loss'], train_results['accuracy']))
#         print('Val. Loss: {} | Val. Acc: {}'.format(val_results['loss'], val_results['accuracy']))
#         print('Train recall: {} | Train precision: {}'.format(train_results['recall'], train_results['precision']))
#         print('Val. recall: {} | Val. precision: {}'.format(val_results['recall'], val_results['precision']))
#
#         #Scheduler
#         #scheduler.step()
#
#     return train_metrics, val_metrics
#
# directory_checkpoint = f"results/checkpoints/25/"
# directory = f"results/25/"
#
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
#
# print('Training and evaluation for {} epochs...'.format(5))
# train_metrics, val_metrics = train_and_evaluate(5, model, optimizer, criterion, train_iter, valid_iter, 10, "", use_bert=False)
# print(train_metrics)

# import pandas as pd
#
# metrics = pd.read_csv("val_metrics.csv")
# print(metrics["recall"][0])

### FEATURES ###
text = "HALLO hallo how are you? 😐😐😥 #hashtag #hash"

import emoji
import re


#Emooji
print(emoji.emoji_count(text))

#Hashtags
print(len(re.findall("#",text)))

#Upper case
print(len([x for x in text.split() if x.isupper()]))

#Stopwords
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
print(len([x for x in text.split() if x in stopwords]))

#
