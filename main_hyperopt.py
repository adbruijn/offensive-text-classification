import click

from main import ex as experiment

import pandas as pd
import numpy as np

from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
(
import pickle

#Sources:
#Sacred and Hyperopt Example: https://github.com/gereleth/kaggle-telstra/blob/master/Automatic%20model%20tuning%20with%20Sacred%20and%20Hyperopt.ipynb
#Click: https://palletsprojects.com/p/click/

# $ python main_hyperopt.py --model_name="MLP"
URL = 

@click.command()
@click.option('--model_name', default="MLP", help="Model name (LSTM, MLP, CNN, LSTMAttention)")
@click.option('--max_evals', default=4, help="Maximum evaluations for the optimisation")
@click.option('--num_epochs', default=50, help="Maximum number of epochs")
@click.option('--embedding_file', default='data/GloVe/glove.twitter.27B.200d.txt', help="Default datapath data/GloVe/glove.twitter.27B.200d.txt'")
#@click.option('--early_stopping_criteria', default=20, help="Stopping critera after a certain number of epochs the validation loss doesn't increase")
def optimize(model_name, max_evals, num_epochs, embedding_file, early_stopping_criteria):
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

    space = {
        'learning_rate': hp.loguniform('learning_rate', -5, -3),
        #'train_bs': hp.quniform('train_bs', 30, 150, 20)
    }

    #Objective
    def objective(params):
        config = {'model_name':model_name, 'num_epochs':num_epochs, 'num_epochs':num_epochs}

        for (key, value) in params.items():
            print(key); print(value)
            config[key] = float(value)

        run = experiment.run(config_updates=config)
        print(config)
        print("Result:", run.result)
        result = run.result

        return run.result

    try:
        trials = pickle.load(open('results/'+model_name+'_results.pkl', 'rb'))
        print("Found saved Trials! Loading...")
        print("Rerunning from {} trials to add {} trial(s)".format(len(trials.trials),max_evals-len(trials.trials)))
    except:
        trials = Trials()
        trials = MongoTrials(URL, exp_key='exp1')
        print("No previous trial found, starting new trials...")

    #Optimisation
    best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    pickle.dump(trials, open('results/'+model_name+'_results.pkl', "wb"))

    for trial in trials.trials:
        print(trial['result'])

    print("Best:", best)

if __name__ == '__main__':
    optimize()
