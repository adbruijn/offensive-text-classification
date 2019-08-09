from utils import generate_features
from data_loader import encode_label, load_data_features, get_data_features, get_data
import numpy as np
from torch.utils.data import (DataLoader, TensorDataset)
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

max_seq_len = 45
embedding_file = 'data/fastText/wiki-news-300d-1M.vec'
batch_size = 100
hidden_dim = 64

embedding_dim, vocab_size, embedding_matrix, train_loader, val_dataloader, test_dataloader = get_data_features(max_seq_len, embedding_file, batch_size)

class MLP(nn.Module):
    def __init__(self):
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

model = MLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):

    for i, data in enumerate(train_loader):
        tweet, features, labels = data

        labels = torch.autograd.Variable(labels).long()

        optimizer.zero_grad()

        outputs = model(tweet, features)
        predictions = torch.argmax(outputs, dim=1).detach().numpy()

        print(labels.numpy())
        print(predictions)

        correct = np.sum(predictions == labels.numpy())
        accuracy = correct / len(predictions)
        print(accuracy)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
