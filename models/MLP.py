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
            embedding_matrix: Pre-trained word embeddings matrix
            embedding_dim: Embedding dimension of the word embeddings
            vocab_size: Dimension of the vocabulary
            hidden_dim: Dimension of the hiddden states
            dropout: Dropout probability
            output_dim: Number of output classes (Subtask A: 2 = (OFF, NOT))
        """

        super(MLP, self).__init__()

        #Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=False)

        #Layer(s)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, int(hidden_dim/2))

        #Dropout
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

    def __init__(self, embedding_matrix, embedding_dim, vocab_size, hidden_dim, dropout, output_dim):

        super(MLP, self).__init__()

        #Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=False)

        #Layers
        self.fc1_embedding = nn.Linear(embedding_dim, hidden_dim)
        self.fc2_embedding = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_embedding = nn.Linear(hidden_dim, int(hidden_dim/2))

        self.fc1 = nn.Linear(in_features=14, out_features=14)
        self.fc2 = nn.Linear(in_features=14, out_features=6)
        self.fc3 = nn.Linear(in_features=6, out_features=6)

        self.output = nn.Linear(int(hidden_dim/2)+6, 2)
        #self.output = nn.Linear(int(hidden_dim/2), 2)
        #self.output = nn.Linear(6, 2)

    def forward(self, tweet, features):

        embedded = torch.mean(self.word_embeddings(tweet), dim=1)
        embedded = embedded.view(embedded.size(0), -1)

        x1 = self.fc1_embedding(embedded)
        x1 = self.fc2_embedding(x1)
        x1 = self.fc3_embedding(x1)

        x2 = self.fc1(features)
        x2 = self.fc2(x2)
        x2 = self.fc3(x2)

        print("Size X1", x1.size())
        print("Size X2", x2.size())

        x3 = torch.cat((x1, x2), dim=1)

        print("Size X3", x3.size())

        x3 = self.output(x3)

        return x3
