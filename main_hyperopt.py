import click

from main import ex as experiment
import pandas as pd
import numpy as np

from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
import pickle

#Sources:
#Sacred and Hyperopt Example: https://github.com/gereleth/kaggle-telstra/blob/master/Automatic%20model%20tuning%20with%20Sacred%20and%20Hyperopt.ipynb
#Click: https://palletsprojects.com/p/click/

# $ python main_hyperopt.py --model_name="MLP"
URL = 'mongodb://localhost:27017/hyperopt/jobs'

@click.command()
@click.option('--model_name', default="MLP", help="Model name (LSTM, MLP, CNN, LSTMAttention)")
@click.option('--max_evals', default=100, help="Maximum evaluations for the optimisation")
@click.option('--num_epochs', default=500, help="Maximum number of epochs")
@click.option('--embedding_file', default='data/GloVe/glove.twitter.27B.100d.txt', help="Default datapath data/GloVe/glove.twitter.27B.200d.txt'")

def optimize(model_name, max_evals, num_epochs, embedding_file):
    #Space
    # hidden_dim = {'hidden_dim': hp.choice("hidden_dim", np.arange(5, 101, 5))}
    # num_layers = {'num_layers': hp.quniform('num_layers', 1, 10)}
    # bidirectional = {'bidirectional': hp.choice("bidirectional", [True, False])}
    # dropout = {'dropout': hp.quniform('dropout', 0.01, 0.2)}
    # max_seq_length = {'max_seq_length': hp.choice("hidden_dim", np.arange(40,71,5))}
    # learning_rate = {'learning_rate': hp.loguniform('learning_rate', -5, 0)} #Learning rate for the model (default=3e-5)
    #dropout = {'dropout': } #Dropout percentage
    #embedding_file = 'data/GloVe/glove.twitter.27B.200d.txt' #Embedding file
    #use_mongo = False

    if model_name=="MLP":
        #MLP
        space = {
            'learning_rate': hp.quniform('learning_rate', 0.00001, 0.001, 0.000001),
            'train_bs': hp.quniform('train_bs', 30, 150, 20),
            'hidden_dim': hp.quniform("hidden_dim", 50,  250, 10),
            'dropout': hp.quniform('dropout', 0.01, 0.5, 0.005),
            'max_seq_length': hp.quniform("max_seq_length", 40, 80, 5)
        }
    elif model_name in "LSTM":
        #LSTM / LSTMAttention
        space = {
            'learning_rate': hp.quniform('learning_rate', 0.00001, 0.001, 0.000001),
            'train_bs': hp.quniform('train_bs', 30, 150, 20),
            'hidden_dim': hp.quniform("hidden_dim", 50,  250, 10),
            'dropout': hp.quniform('dropout', 0.01, 0.5, 0.005),
            'max_seq_length': hp.quniform("max_seq_length", 40, 80, 5),
            'num_layers': hp.quniform("max_seq_length", 2, 20, 1)
        }
    # elif model_name in "BERT":
    #     #BERT
    #     space = {
    #         'learning_rate': hp.quniform('learning_rate', 0.00001, 0.001, 0.000001),
    #         'train_bs': hp.quniform('train_bs', 30, 150, 20),
    #         'max_seq_length': hp.quniform("max_seq_length", 40, 80, 5),
    #         'dropout': hp.quniform('dropout', 0.01, 0.5, 0.005),
    #         'num_layers': hp.quniform("max_seq_length", 2, 20, 1)
    #     }
    # elif model_name=="CNN":
    #     space = {
    #         'dropout': hp.quniform('dropout', 0.01, 0.5, 0.005),
    #         #'filter_sizes': hp.quniform("filter_sizes", 40, 80, 5),
    #         #'filter_dim': hp.quniform("filter_dim", 2, 20, 1)
    #     }

    #Objective
    def objective(params):
        config = {'model_name':model_name, 'num_epochs':num_epochs}

        for (key, value) in params.items():
            config[key] = value
            print("Parameter: {} | Value: {}".format(key, value))

        run = experiment.run(config_updates=config)

        return run.result

    try:
        trials = pickle.load(open('results/'+model_name+'_results2.pkl', 'rb'))
        print("Load {} Trials...".format(model_name))
        print("Rerunning from {} trials to add {} trial(s)".format(len(trials.trials),max_evals-len(trials.trials)))
    except:
        trials = Trials()
        #trials = MongoTrials('mongo://localhost:27017/hyperopt/jobs', exp_key='exp1')
        print("No previous trial found, starting new trials...")

    #Optimisation
    best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    pickle.dump(trials, open('results/'+model_name+'_results2.pkl', "wb"))

    for trial in trials.trials:
        print(trial['result'])

    print("Best:", best)

if __name__ == '__main__':
    optimize()
