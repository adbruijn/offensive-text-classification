# Offensive Text Classification

## Description
Classifies offensive text using different machine learning models: MLP, LSTM/LSTM (with and without attention), CNN and BERT.
GloVe word vectors are used as input feature for the models. The features for the model BERT are extracted with the function convert_examples_to_features.

## Data
There are 3 datasets in the folder data: training, validation and test set. Each of datasets contain the columns: text and label.

## Installation
Install the required packages with requirements.txt.
```
pip install -r requirements.txt
```

## Usage
Run from the command line
```python
python main.py
```

Sacred is used for managing experiments. To choose other parameters you have to use 'with' and then specify the other parameter.
Example command run from the command line for choosing another model
```python
python main.py with model_name="BERT" output_dim=2
```

You can either choose to store the experiments locally or on a MongoDB, or both.
If the results of the experiments are stored in a MongoDB. Omniboard, a visualuation board can be used to visualise the results. Locally the experiments will be stored in the 'results' folder.

OMNIBOARD
Install: ```npm install -g omniboard```

Open a terminal:
```
sudo mongod
```

Open another terminal:
mongo

```show dbs
> show dbs
admin          0.000GB
config         0.000GB
local          0.000GB
lstm           0.001GB
```

Drop a database:
```
use lstm
db.dropDatabase()
```
Create a database:
```
use DATABASE_NAME
```
The created database is not present when using
```
show dbs
```
There has to be at least one document in the database.
```
db.movie.insert({"name":"hoi"})
```

Run:
```
omniboard -m hostname:port:database
```
for example
```
> omniboard -m localhost:27017:model
Omniboard is listening on port 9000!
```
Go to the page http://localhost:9000 to track the results.

Shutdown the server
```
> use admin
> db.shutdownServer()
```
