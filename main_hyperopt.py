import click

from main import ex as experiment

import pandas as pd
import numpy as np

from hyperopt import fmin, tpe, hp, Trials

#Sources:
#Sacred and Hyperopt Example: https://github.com/gereleth/kaggle-telstra/blob/master/Automatic%20model%20tuning%20with%20Sacred%20and%20Hyperopt.ipynb
#Click: https://palletsprojects.com/p/click/

@click.command()
@click.option('--model_name', default="MLP", help="Model name (LSTM, MLP, CNN, LSTMAttention)")
@click.option('--max_evals', default="10", help="Maximum evaluations for the optimisation")
def run(model_name):
    #Space
    space = {'hidden_dim': hp.choice("hidden_dim", [4, 16, 32, 64, 128, 256])}

    #Objective
    def objective(params):
        config = {'model_name':model_name}
        for (key, value) in params.items():
            config[key] = int(value)
        #print(config)
        run = experiment.run(config_updates=config)
        print("Result:", run.result)
        return run.result

    #Optimisation
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)
    print("Best", best)

if __name__ == '__main__':
    run()
