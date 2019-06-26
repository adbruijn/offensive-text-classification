import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pack_sequence

class MLP(nn.Module):

    def __init__(self, embedding_matrix, embedding_dim, vocab_size, hidden_dim, dropout, output_dim):

        """
        Args:
            embedding_matrix: Pre-trained word embeddings
            embedding_dim: Embedding dimension of the word embeddings
            vocab_size: Size of the vocabulary
            hidden_dim: Size hiddden state
            dropout: Dropout probability
            output_dim: Output classes (Subtask A: 2 = (OFF, NOT))
        """

        super(MLP, self).__init__()

        #Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=False)

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.dropout = nn.Dropout(dropout)

        #Linear layer
        self.output = nn.Linear(int(hidden_dim/2), output_dim)

    def forward(self, x):

        embedded = torch.mean(self.word_embeddings(x), dim=1)
        embedded = embedded.view(embedded.size(0), -1)

        x = F.relu(self.linear1(embedded))
        x = F.relu(self.linear2(self.dropout(x)))
        x = F.relu(self.linear3(x))
        x = self.output(x)

        return x

class MLP_Features(nn.Module):

    def __init__(self, embedding_matrix, embedding_dim, vocab_size, hidden_dim, features_dim, dropout, output_dim):

        """
        Args:
            embedding_matrix: Pre-trained word embeddings
            embedding_dim: Embedding dimension of the word embeddings
            vocab_size: Size of the vocabulary
            hidden_dim: Size hiddden state
            dropout: Dropout probability
            output_dim: Output classes (Subtask A: 2 = (OFF, NOT))
        """

        super(MLP_Features, self).__init__()

        #Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=False)

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, int(hidden_dim/2))

        self.linear_f = nn.Linear(features_dim, int(hidden_dim/2))

        self.dropout = nn.Dropout(dropout)

        #Linear layer
        self.output = nn.Linear(int(hidden_dim/2), output_dim)

    def forward(self, x, features):

        print("x:", x.size(0))

        embedded = torch.mean(self.word_embeddings(x), dim=1)
        embedded = embedded.view(embedded.size(0), -1)

        print("\n")
        print("Features:", features.size(0))
        print("Embedded:", embedded.size(0))
        print("\n")

        x = F.relu(self.linear1(embedded))
        # print("linear1:", x.size(0))

        x = F.relu(self.linear2(self.dropout(x)))
        # print("linear2:", x.size(0))

        x = F.relu(self.linear3(x))
        # print("linear3:", x.size(0))
        #
        # print("f0:", features.size(0))
        # print("f1:", features.size(1))
        features = self.linear_f(features.view(features.size(0), -1))


        #combined = torch.cat((x, features), dim=1)
        print(x.size(0))
        print(x.size(1))

        print(features.size(0))
        print(features.size(1))

        combined = torch.cat((x.view(x.size(0), -1),
                          features.view(features.size(0), -1)), dim=1)

        out = self.output(combined)

        return out
