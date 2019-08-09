import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class LSTM(nn.Module):

    def __init__(self, embedding_matrix, embedding_dim, vocab_size, hidden_dim, dropout, num_layers, bidirectional, output_dim):

        """
        Args:
            embedding_matrix: Pre-trained word embeddings matrix
            embedding_dim: Embedding dimension of the word embeddings
            vocab_size: Dimension of the vocabulary
            hidden_dim: Dimension of the hiddden states
            dropout: Dropout probability
            num_layers: Number of layers of the LSTM
            bidirectional: Bidiredctional
            output_dim: Number of output classes (Subtask A: 2 = (OFF, NOT))
        """

        super(LSTM, self).__init__()
        self.num_layers =  num_layers
        self.hidden_dim =  hidden_dim
        self.bidirectional =  bidirectional

        #Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=False)

        #Dropout
        self.dropout = dropout

        #LSTM layer(s)
        if(self.bidirectional):
            self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2 , num_layers, dropout=self.dropout, bidirectional=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=self.dropout)

        #Linear layer
        self.output = nn.Linear(in_features=hidden_dim, out_features=output_dim)


    def forward(self, X):

        #Word embeddings
        embedded = self.word_embeddings(X)
        embedded = embedded.permute(1,0,2)

        #Batch size
        batch_size = X.size(0)

        #Initial hidden state
        if(self.bidirectional):
            h0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim // 2))
        else:
            h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim))

        #Forward state
        output, (hidden_state, cell_state) = self.lstm(embedded, (h0, c0))

        x = self.output(output[-1])

        return x
