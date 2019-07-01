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

class BertNonLinear(nn.Module):
    def __init__(self, dropout, output_dim):
        """
        Args:
            dropout: Dropout probability
            output_dim: Output dimension (number of labels)
        """

        super(BertNonLinear, self).__init__()
        self.output_dim = output_dim
        self.dropout = dropout

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 768)
        self.linear3 = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        x = self.linear1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))

        return x

class BertNorm(nn.Module):
    def __init__(self, dropout, output_dim):
        """
        Args:
            dropout: Dropout probability
            output_dim: Output dimension (number of labels)
        """

        super(BertNorm, self).__init__()
        self.output_dim = output_dim
        self.dropout = dropout

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Sequential(
            nn.Linear(768,768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Linear(768, output_dim)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        x = self.fc(pooled_output)

        return x

class BertLinearFreeze(nn.Module):
    def __init__(self, hidden_dim, dropout, output_dim):
        """
        Args:
            hidden_dim: Size hiddden state
            dropout: Dropout probability
            output_dim: Output dimension (number of labels)
        """

        super(BertLinearFreeze, self).__init__()
        self.output_dim = output_dim
        self.dropout = dropout

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = False
            print(param)
            print(param.requires_grad)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

        self.linear1 = nn.Linear(768, output_dim)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        x = self.linear1(pooled_output)

        return x

class BertLinearFreezeEmbeddings(nn.Module):
    def __init__(self, hidden_dim, dropout, output_dim):
        """
        Args:
            hidden_dim: Size hiddden state
            dropout: Dropout probability
            output_dim: Output dimension (number of labels)
        """

        super(BertLinearFreezeEmbeddings, self).__init__()
        self.output_dim = output_dim
        self.dropout = dropout

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for name, param in self.bert.named_parameters():
            if name.startswith('embeddings'):
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

        self.linear1 = nn.Linear(768, output_dim)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        x = self.linear1(pooled_output)

        return x

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
        self.linear4 = nn.Linear(768, output_dim)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        #x = self.dropout(pooled_output)

        x = self.linear4(pooled_output)
        # x = self.linear2(x)
        # x = self.linear3(x)

        # x = self.relu(self.linear1(pooled_output))
        # x = self.dropout(x)
        # x = self.relu(self.linear2(x))
        # x = self.relu(self.linear3(x))

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
