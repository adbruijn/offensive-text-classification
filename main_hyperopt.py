import click

from main import ex as experiment

import pandas as pd
import numpy as np

from hyperopt import fmin, tpe, hp, Trials

import pickle

#Sources:
#Sacred and Hyperopt Example: https://github.com/gereleth/kaggle-telstra/blob/master/Automatic%20model%20tuning%20with%20Sacred%20and%20Hyperopt.ipynb
#Click: https://palletsprojects.com/p/click/

# $ python main_hyperopt.py --model_name="MLP"

@click.command()
@click.option('--model_name', default="MLP", help="Model name (LSTM, MLP, CNN, LSTMAttention)")
@click.option('--max_evals', default=10, help="Maximum evaluations for the optimisation")
@click.option('--num_epochs', default=100, help="Maximum number of epochs")
@click.option('--embedding_file', default='data/GloVe/glove.twitter.27B.200d.txt', help="Default datapath data/GloVe/glove.twitter.27B.200d.txt'")

def run(model_name, max_evals, num_epochs, embedding_file):
    #Space
    hidden_dim = {'hidden_dim': hp.choice("hidden_dim", np.arange(5, 101, 5))}
    num_layers = {'num_layers': hp.quniform('num_layers', 1, 10)}
    bidirectional = {'bidirectional': hp.choice("bidirectional", [True, False])}
    dropout = {'dropout': hp.quniform('dropout', 0.01, 0.2)}
    max_seq_length = {'max_seq_length': hp.choice("hidden_dim", np.arange(40,71,5))}
    learning_rate = {'learning_rate': hp.loguniform('learning_rate', -5, 0)} #Learning rate for the model (default=3e-5)
    #dropout = {'dropout': } #Dropout percentage
    #embedding_file = 'data/GloVe/glove.twitter.27B.200d.txt' #Embedding file
    #use_mongo = False

    #Objective
    def objective(params):
        config = {'model_name':model_name, 'num_epochs':num_epochs}

        for (key, value) in params.items():
            config[key] = int(value)

        run = experiment.run(config_updates=config)
        print("Result:", run.result)
        result = run.result

        return run.result

    try:
        trials = pickle.load(open('results/'+model_name+'_results.pkl', 'rb'))
        print("Found saved Trials! Loading...")
        print("Rerunning from {} trials to add {} trial(s)".format(len(trials.trials),max_evals-len(trials.trials)))
    except:
        trials = Trials()
        print("No previous trial found, starting new trials...")

    #Optimisation
    best = fmin(fn=objective, space=hidden_dim, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    pickle.dump(trials, open('results/'+model_name+'_results.pkl', "wb"))
    print(trials.trials)
    print(trials)
    print(best)

if __name__ == '__main__':
    run()
