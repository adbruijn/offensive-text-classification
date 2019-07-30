### HIDDEN STATES ###
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Text
from data_loader import load_data, clean_data

subtask = 'a'
texts = ["Here is the sentence I want embeddings for.",
"After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."]

train, val, test = load_data(subtask)

#Clean data
X_train, y_train = clean_data(train)
X_val, y_val = clean_data(val)
X_test, y_test = clean_data(test)

max_seq_length = 40
def get_features(texts):

    col_names = ["tokenized_text","tokens_tensor","segments_tensors"]
    features = pd.DataFrame(columns=col_names)

    for text in texts:
        marked_text = "[CLS] " + text + " [SEP]"

        #print("Text: ", text)

        # Tokenization
        tokenized_text = tokenizer.tokenize(marked_text)
        #print("Tokenized text: ", tokenized_text)

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Segement IDs
        segments_ids = [1] * len(tokenized_text)

        # Padding

        padding = [0] * (max_seq_length - len(indexed_tokens))

        indexed_tokens += padding
        segments_ids += padding

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        input_features = {'tokenized_text': tokenized_text, 'tokens_tensor':tokens_tensor, 'segments_tensors':segments_tensors}
        features.loc[len(features)] = input_features

    return features

features = get_features(X_test)
# val_features = get_features(X_val)
# test_features = get_features(X_test)
#print(features)

### Predict hidden states features for each layer ###
model.eval()

with torch.no_grad():
    # all_encoded_layers = []

    # for i  in range(len(features.index)):
        #print(i)
    encoded_layers, _ = model(features['tokens_tensor'][0], features['segments_tensors'][0])
        # all_encoded_layers.append(encoded_layers)

#
# def get_embedding(encoded_layers):
#     token_embeddings = []
#     for token_i in range(max_seq_length):
#
#         hidden_layers = []
#
#         for layer_i in range(len(encoded_layers)):
#             vec = encoded_layers[layer_i][0][token_i]
#
#             hidden_layers.append(vec)
#
#     token_embeddings.append(hidden_layers)
#
#     ### First Layer ###
#     first_layer = torch.mean(encoded_layers[0], 1)
#     # print(len(first_layer), len(first_layer[0]))
#
#     ### Second-to-Last Hidden ###
#     second_to_last = torch.mean(encoded_layers[11], 1)
#
#     # print(len(second_to_last), len(second_to_last[0]))
#
#     ### Sum Last Four Hidden ###
#     # token_last_four_sum = []
#     #
#     # for token in token_embeddings:
#     #     sum_vec = torch.sum(torch.stack(token)[-4:], 0)
#     #     token_last_four_sum.append(sum_vec)
#     #
#     # print(len(token_last_four_sum), len(token_last_four_sum[0]))
#     token_last_four_sum = [torch.sum(torch.stack(token)[-4:], 0) for token in token_embeddings]
#
#     ### Concat Last Four Hidden ###
#     # token_last_four_cat = []
#     # for token in token_embeddings:
#     #     cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), 0)
#     #     token_last_four_cat.append(cat_vec)
#     #
#     # print(len(token_last_four_cat), len(token_last_four_cat[0]))
#     token_last_four_cat = [torch.cat((token[-1], token[-2], token[-3], token[-4]), 0) for token in token_embeddings]
#
#     ### Sum All 12 Layers ###
#     # token_sum_all = []
#     #
#     # for token in token_embeddings:
#     #     sum_vec = torch.sum(torch.stack(token)[0:], 0)
#     #     token_sum_all.append(sum_vec)
#     #
#     # print(len(token_sum_all), len(token_sum_all[0]))
#     token_sum_all = [torch.sum(torch.stack(token)[0:], 0) for token in token_embeddings]
#
#     return first_layer, second_to_last, token_last_four_sum, token_last_four_cat, token_sum_all
#
# first, second_to_last, last_four_sum, last_four_cat, sum_all = get_embedding(all_encoded_layers[0])
#
# col_names = ['first', 'second_to_last', 'last_four_sum', 'last_four_cat', 'sum_all']
# embeddings = pd.DataFrame(columns=col_names)
#
# for i, encoded_layer in enumerate(all_encoded_layers):
#     print(i)
#     first, second_to_last, last_four_sum, last_four_cat, sum_all = get_embedding(encoded_layer)
#
#     input_embeddings = {'first': first, 'second_to_last':second_to_last, 'last_four_sum':last_four_sum, 'last_four_cat':last_four_cat, 'sum_all':sum_all}
#     embeddings.loc[len(embeddings)] = input_embeddings
#
# # print(embeddings)
#
# embeddings.to_csv("data/test_embeddings_bert.csv")
