import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class BertLinear(nn.Module):
    def __init__(self, hidden_dim, dropout, output_dim):
        """
        Args:
            hidden_dim: Size hiddden state
            dropout: Dropout probability
            output_dim: Output dimension (number of labels)
        """

        super(BertLinear, self).__init__()
        self.output_dim = output_dim
        self.dropout = dropout

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

        self.linear1 = nn.Linear(768, hidden_dim) #self.bert.config.hidden_size = 768
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        x = self.relu(self.linear1(pooled_output))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))

        return x

class BertLSTM(nn.Module):
    def __init__(self, hidden_dim, dropout, bidrectional, output_dim):
        """
        Args:
            dropout: Dropout probability
            output_dim: Output dimension (number of labels)
        """

        super(BertLSTM, self).__init__()
        self.output_dim = output_dim
        self.dropout = dropout

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(768, hidden_dim, bidirectional) #self.bert.config.hidden_size = 768

        if bidrectional:
            self.output = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        encoded_layers = encoded_layers.permute(1, 0 ,2)

        output, (hidden_state, cell_state) = self.lstm(encoded_layers)

        out = torch.cat((hidden_state[0], hidden_state[1]), dim=1)

        out = self.dropout(out)

        x = self.output(out)

        return x

# class Bert(nn.Module):
#     def __init__(self, dropout, output_dim):
#         """
#         Args:
#             dropout: Dropout probability
#             output_dim: Output dimension (number of labels)
#         """
#
#         super(Bert, self).__init__()
#         self.output_dim = output_dim
#         self.dropout = dropout
#
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#
#         self.drop = nn.Dropout(dropout)
#
#         self.linear1 = nn.Linear(768, 100)
#         self.batch_norm = nn.BatchNorm1d(100)
#         #self.relu = nn.ReLU(inplace=True)
#         self.linear2 = nn.Linear(100, 2)
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#
#         logits = self.linear2(self.batch_norm(self.linear1(encoded_layers)))
#
#         return logits
