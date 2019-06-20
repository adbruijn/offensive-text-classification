import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class LSTMAttention(nn.Module):

    def __init__(self, embedding_matrix, embedding_dim, vocab_size, hidden_dim, dropout, num_layers, bidirectional, output_dim):

        """
        Args:
            embedding_matrix: Pre-trained word embeddings
            hidden_dim: Size hiddden state
            vocab_size: Size vocabulary containing unique words
            embedding_dim: Embedding dimension of the word embeddings
            output_dim: Output classes (Subtask A: 2 = (OFF, NOT))
        """

        super(LSTMAttention, self).__init__()
        self.hidden_dim =  hidden_dim
        self.dropout = dropout

        #Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=False)

        #LSTM layer(s)
        if(self.bidirectional):
            self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2 , num_layers, dropout=self.dropout, bidirectional=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=self.dropout)

        #Linear layer
        self.output = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def attention(self, out, state):

        """
        Use attention to compute soft alignment score between each hidden state and the last hidden state (torch.bmm: batch matrix multiplication)
        """

        hidden = state.squeeze(0)
        attn_weights = torch.bmm(out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden


    def forward(self, X):

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
        output, (hidden_state, cell_state) = self.lstm(embedded, (h0, c0)) #hidden_size?, batch_size, hidden_size
        output = output.permute(1, 0, 2)
        attn_output = self.attention(output, hidden_state)

        x = self.output(attn_output)

        return x
