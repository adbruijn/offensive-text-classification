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

class BertExtractEmbeddings(nn.Module):

    def __init__(self, dropout, output_dim):

        """
        Args:
            embedding_matrix: Pre-trained word embeddings
            embedding_dim: Embedding dimension of the word embeddings
            vocab_size: Size of the vocabulary
            hidden_dim: Size hiddden state
            dropout: Dropout probability
            output_dim: Output classes (Subtask A: 2 = (OFF, NOT))
        """

        super(BertExtractEmbeddings, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, output_dim)
        self.dropout = nn.Dropout(dropout)

    def extract_embedding(enc_layers):

        max_seq_length = len(enc_layers[0][0])
        token_embeddings = []

        for token_i in range(max_seq_length):
            hidden_layers = []

            for layer_i in range(len(enc_layers)):
                vec = enc_layers[layer_i][0][token_i]
                hidden_layers.append(vec)

        token_embeddings.append(hidden_layers)

        first_layer = torch.mean(enc_layers[0], 1)
        # second_to_last = torch.mean(encoded_layers[11], 1)
        # token_last_four_sum = [torch.sum(torch.stack(token)[-4:], 0) for token in token_embeddings] #sum last four hidden layers
        # token_last_four_cat = [torch.cat((token[-1], token[-2], token[-3], token[-4]), 0) for token in token_embeddings]
        # token_sum_all = [torch.sum(torch.stack(token)[0:], 0) for token in token_embeddings]

        return first_layer

    # def extract_bert_embeddings(encoded_layers):
    #     max_seq_length = len(encoded_layers[0][0])
    #     token_embeddings = []
    #
    #     for token_i in range(max_seq_length):
    #         hidden_layers = []
    #
    #         for layer_i in range(len(encoded_layers)):
    #             vec = encoded_layers[layer_i][0][token_i]
    #             hidden_layers.append(vec)
    #
    #         token_embeddings.append(hidden_layers)
    #
    #     return "hoi"
    #     #first_layer = torch.mean(encoded_layers[0], 1)

        #return first_layer


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        with torch.no_grad():
            encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)

            max_seq_length = len(encoded_layers[0][0])
            token_embeddings = []

            for token_i in range(max_seq_length):
                hidden_layers = []

                for layer_i in range(len(encoded_layers)):
                    vec = encoded_layers[layer_i][0][token_i]
                    hidden_layers.append(vec)

                token_embeddings.append(hidden_layers)

            first_layer = torch.mean(encoded_layers[0], 1)

            #first_layer = self.extract_bert_embeddings(encoded_layers)

            #first_layer = self.get_embedding(encoded_layers)
            #first_layer = torch.mean(encoded_layers[0], 1)
            #print(p)

        x = self.classifier(first_layer)

        return x

class BertPooling(nn.Module):

    def __init__(self, dropout, output_dim):

        """
        Args:
            embedding_matrix: Pre-trained word embeddings
            embedding_dim: Embedding dimension of the word embeddings
            vocab_size: Size of the vocabulary
            hidden_dim: Size hiddden state
            dropout: Dropout probability
            output_dim: Output classes (Subtask A: 2 = (OFF, NOT))
        """

        super(BertPooling, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(768, output_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        with torch.no_grad():
            encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
            input = encoded_layers[-1]
            print("Pooled output size", pooled_output.size())
            print(pooled_output)
            mean_encoded_layers = torch.stack(encoded_layers, dim=0).mean(dim=0).mean(dim=0)
            print("Mean encoded layers size:", mean_encoded_layers)

        #x = self.dropout(pooled_output)
        x = self.classifier(mean_encoded_layers)

        return x

class BertTest(nn.Module):

    def __init__(self, dropout, output_dim):

        """
        Args:
            embedding_matrix: Pre-trained word embeddings
            embedding_dim: Embedding dimension of the word embeddings
            vocab_size: Size of the vocabulary
            hidden_dim: Size hiddden state
            dropout: Dropout probability
            output_dim: Output classes (Subtask A: 2 = (OFF, NOT))
        """

        super(BertTest, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(768, output_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        x = self.dropout(pooled_output)
        x = self.classifier(x)

        return x


class BertNonLinear(nn.Module):
    def __init__(self, dropout, output_dim):
        """
        Args:
            dropout: Dropout probability
            output_dim: Output dimension (number of labels)
        Output:
            encoded layers: outputs a list of the full sequences of encoded-hidden-states at the end oof ech attention block
            pooled_output: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
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

        x = self.relu(self.dropout(self.linear1(pooled_output)))
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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        all_encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

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
            #print(param)
            #print(param.requires_grad)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

        self.linear1 = nn.Linear(768, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim , output_dim)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        x = self.linear1(pooled_output)
        x = self.relu(x)
        x = self.linear2(x)

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

        self.linear1 = nn.Linear(768, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        x = self.linear1(pooled_output)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

class BertLinearFreezeEmbeddingsMultiple(nn.Module):
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
        x = self.linear2(x)
        x = self.linear3(x)

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

        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.2)

        self.relu = nn.LeakyReLU()

        self.linear1 = nn.Linear(768, 768) #self.bert.config.hidden_size = 768
        self.linear2 = nn.Linear(768, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        x = self.dropout1(pooled_output)
        x = self.relu(self.linear1(x))
        x = self.dropout2(x)
        x = self.relu(self.linear2(x))
        x = self.dropout3(x)
        x = self.relu(self.linear3(x))
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
