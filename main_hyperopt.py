import click

from main import ex as experiment

import pandas as pd
import numpy as np

from hyperopt import fmin, tpe, hp, Trials

import pickle

#Sources:
#Sacred and Hyperopt Example: https://github.com/gereleth/kaggle-telstra/blob/master/Automatic%20model%20tuning%20with%20Sacred%20and%20Hyperopt.ipynb
#Click: https://palletsprojects.com/p/click/

@click.command()
@click.option('--model_name', default="MLP", help="Model name (LSTM, MLP, CNN, LSTMAttention)")
@click.option('--max_evals', default=1, help="Maximum evaluations for the optimisation")
@click.option('--num_epochs', default=10, help="Maximum number of epochs")

def run(model_name, max_evals, num_epochs):
    #Space
    space = {'hidden_dim': hp.choice("hidden_dim", [4, 16, 32, 64, 128, 256])}
    num_layers = {'num_layers': hp.choice()}
    bidirectional = {'bidirectional': hp.choice("bidirectional", [True, False])}
    dropout = {'dropout': }

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
    best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    pickle.dump(trials, open('results/'+model_name+'_results.pkl', "wb"))
    print(trials.trials)
    print(trials)
    print(best)

if __name__ == '__main__':
    run()
