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
    def __init__(self, dropout, output_dim):
        """
        Args:
            dropout: Dropout probability
            output_dim: Output dimension (number of labels)
        """

        super(BertLinear, self).__init__()
        self.output_dim = output_dim
        self.dropout = dropout

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.drop = nn.Dropout(dropout)

        self.linear1 = nn.Linear(768, 100) #self.bert.config.hidden_size = 768
        self.linear2 = nn.Linear(100, output_dim)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        output = self.linear1(self.drop(pooled_output))
        logits = self.linear2(output)

        return logits

class BertLSTM(nn.Module):
    def __init__(self, dropout, output_dim):
        """
        Args:
            dropout: Dropout probability
            output_dim: Output dimension (number of labels)
        """

        super(BertLSTM, self).__init__()
        self.output_dim = output_dim
        self.dropout = dropout

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(768, 100, bidirectional=True) #self.bert.config.hidden_size = 768
        self.linear = nn.Linear(100 * 2, output_dim)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        encoded_layers = encoded_layers.permute(1, 0 ,2)

        output, (hidden_state, cell_state) = self.lstm(encoded_layers)

        out = torch.cat((hidden_state[0], hidden_state[1]), dim=1)

        out = self.drop(out)

        logits = self.linear(out)

        return logits
