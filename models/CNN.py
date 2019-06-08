import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class CNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, weights, output_dim, filter_sizes, filter_dim, dropout):

        """
        Arguments:
            vocab_size: Size vocabulary containing unique words
            embedding_dim: Embedding dimension of the word embeddings
            weights: Pre-trained word embeddings

            output_dim: Output classes (Subtask A: 2 = (OFF, NOT))
            filter_dim:
            filter_sizes: List containing 3 different filter sizes.
            dropout: Dropout probability

        """

        super(CNN, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)

        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, out_channels=filter_dim, kernel_size = (ks, embedding_dim)) for ks in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_dim, output_dim)

    def forward(self, X):

        """
        Parameters:
            x: Input of shape = (batch_size, num_sequences)
            batch_size: Batch size

        Returns:
            Output: Linear layer contains the logits for the classes
            logits.size() = (batch_size, output_size)
        """

        embedded = self.word_embeddings(X)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded).squeeze(3)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = torch.cat(pooled, dim=1)
        out = self.dropout(cat)
        out = self.fc(out)

        return out
