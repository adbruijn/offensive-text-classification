import torch
from torch import nn

from pytorch_pretrained_bert import BertConfig, BertForSequenceClassification, BertModel
import models

dropout = 0.1
output_dim = 2
hidden_size = 100

input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

model = models.BertLSTM(dropout, output_dim)
print(model)

logits = model(input_ids, token_type_ids, input_mask)

print(logits)
