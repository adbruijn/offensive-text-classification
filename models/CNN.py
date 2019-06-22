import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class CNN(nn.Module):

    def __init__(self, embedding_matrix, embedding_dim, vocab_size, dropout, filter_sizes, output_dim):

        """
        Args:
            embedding_matrix: Pre-trained word embeddings
            embedding_dim: Embedding dimension of the word embeddings
            vocab_size: Size of the vocabulary
            filter_sizes: List containing 3 different filter sizes
            dropout: Dropout probability
            output_dim: Output classes (Subtask A: 2 = (OFF, NOT))

        """

        super(CNN, self).__init__()

        #Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=False)

        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, out_channels=embedding_dim, kernel_size = (ks, embedding_dim)) for ks in filter_sizes])
        self.dropout = nn.Dropout(dropout)

        #Linear layer
        self.output = nn.Linear(len(filter_sizes) * embedding_dim, output_dim)

    def forward(self, X):

        embedded = self.word_embeddings(X)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded).squeeze(3)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = torch.cat(pooled, dim=1)
        x = self.dropout(cat)
        x = self.output(x)

        return x
