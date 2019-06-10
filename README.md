# Offensive Text Classification

## Description
Classify offensive text using different machine learning models.

## Usage

Run from the command line
```python
python main.py
Example for chooising a different model: python main.py with model_name="BERT"
```

You can either choose to store the experiments locally or on a MongoDB.

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
