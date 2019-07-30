import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class LinearClassifier(nn.Module):

    def __init__(self, num_labels, input_features):

        super(LinearClassifier, self).__init__()

        self.linear = nn.Linear(input_features, num_labels)

    def forward(self, features):

        x = self.linear(features)

        return x

model = LinearClassifier(2, 768)


#Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Text
text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
marked_text = "[CLS] " + text + " [SEP]"
print("Text: ", text)

# Tokenization
tokenized_text = tokenizer.tokenize(marked_text)
print("Tokenized text: ", tokenized_text)

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Segement IDs
segments_ids = [1] * len(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)

    #batch_size, max_len, feat_dim = encoded_layers.shape
    # valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda')

batch_size = len(encoded_layers[1])
batch_token_embeddings = []

for batch_i in range(batch_size):

    token_embeddings = []
    for token_i in range(len(tokenized_text)):

        hidden_layers = []

        for layer_i in range(len(encoded_layers)):
            vec = encoded_layers[layer_i][batch_i][token_i]

            hidden_layers.append(vec)

        token_embeddings.append(hidden_layers)
    batch_token_embeddings.append(token_embeddings)

### First Layer ###
first_layer = torch.mean(encoded_layers[0], 1)
print(len(first_layer), len(first_layer[0]))

### Second-to-Last Hidden ###
second_to_last = torch.mean(encoded_layers[11], 1)
print(len(second_to_last), len(second_to_last[0]))

# ### Last Hidden ###
# last_layer = torch.mean(encoded_layers[12], 1)
# # print(last_layer)
# #print(len(last_layer), len(last_layer[0]))

### Sum Last Four Hidden ###
token_vecs_sum = []

for token in token_embeddings:
    sum_vec = torch.sum(torch.stack(token)[-4:], 0)
    token_vecs_sum.append(sum_vec)

print(len(token_vecs_sum), len(token_vecs_sum[0]))

### Concat Last Four Hidden ###
token_vecs_cat = []
for token in token_embeddings:
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), 0)
    token_vecs_cat.append(cat_vec)

print(len(token_vecs_cat), len(token_vecs_cat[0]))

### Sum All 12 Layers ###
token_vecs_sum_all = []

for token in token_embeddings:
    sum_vec = torch.sum(torch.stack(token)[0:], 0)
    token_vecs_sum_all.append(sum_vec)

print(len(token_vecs_sum_all), len(token_vecs_sum_all[0]))

### Sentence Vectors ###
concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # [number_of_tokens, 3072]
print(len(concatenated_last_4_layers), len(concatenated_last_4_layers[0]))



print(batch_token_embeddings)
