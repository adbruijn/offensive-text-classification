import torch
from torch import nn

from pytorch_pretrained_bert import BertConfig, BertForSequenceClassification, BertModel
import models

# output_dim = 2
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", output_dim)
# print(model)
#
# for p in model.bert.embeddings.parameters():
#     p.requires_grad = False

bert = BertModel.from_pretrained('bert-base-uncased')
for name, param in bert.named_parameters():
    if name.startswith('embeddings'):
        param.requires_grad = False

# dropout = 0.1
# output_dim = 2
# hidden_size = 100
#
# input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
# input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
# token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#
# linear_model = models.BertLinear(dropout, output_dim)
# print("Model - Linear")
# logits = linear_model(input_ids, token_type_ids, input_mask)
# print("Logits Linear: \n", logits)
#
# lstm_model = models.BertLSTM(dropout, output_dim)
# print("Model - LSTM")
# logits = lstm_model(input_ids, token_type_ids, input_mask)
# print("Logits LSTM: \n", logits)
#
# linear_extra = models.Bert(dropout, output_dim)
# print("Model - LSTM")
# logits = linear_extra(input_ids, token_type_ids, input_mask)
# print("Logits LSTM: \n", logits)
