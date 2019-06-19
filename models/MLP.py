import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pack_sequence

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, hidden_dim, dropout, output_dim):
        super(MLP, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=False)

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.linear3 = nn.Linear(int(hidden_dim/2), int(hidden_dim/4))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(int(hidden_dim/4), output_dim)

    def forward(self, x):

        embedded = torch.mean(self.word_embeddings(x), dim=1)
        embedded = embedded.view(embedded.size(0), -1)

        x = F.relu(self.linear1(embedded))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.dropout(x)
        x = self.output(x)

        return x
