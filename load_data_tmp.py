subtask = "b"

subtask_name = "subtask_" + subtask

import pandas as pd
from sklearn.model_selection import train_test_split

train_cola = pd.read_csv("data/SemEval/olid-training-v1.0.tsv", delimiter="\t")
test_cola = pd.read_csv("data/SemEval/testset-level"+subtask+".tsv", delimiter="\t")
labels_cola = pd.read_csv("data/SemEval/labels-level"+subtask+".csv", header=None)
labels_cola.columns = ['id', subtask_name]

test = pd.merge(test_cola, labels_cola, on='id')

# Remove duplicates
train_cola = train_cola.drop_duplicates("tweet")
test = test.drop_duplicates("tweet")

#Remove nan in a certain column
train_cola = train_cola.dropna(subset=[subtask_name])
test = test.dropna(subset=[subtask_name])

train, val = train_test_split(train_cola, test_size=0.2, random_state=123)
train.reset_index(drop=True)
val.reset_index(drop=True)

train = train[["tweet",subtask_name]]
val = val[["tweet",subtask_name]]
test = test[["tweet",subtask_name]]

train.columns = ['text', 'label']
val.columns = ['text','label']
test.columns = ['text', 'label']

print(train.head(10))
print(val.head(10))
print(test.head(10))
