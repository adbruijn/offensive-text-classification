from main import ex as experiment

import pandas as pd
import numpy as np

from hyperopt import fmin, tpe, hp, Trials

#Source: https://github.com/gereleth/kaggle-telstra/blob/master/Automatic%20model%20tuning%20with%20Sacred%20and%20Hyperopt.ipynb

#MLP
#run = experiment.run(config_updates = {"model_name":"MLP"}) #Run experiment with another model

#### Tune model with hyperopt ###
#Space
space = {'hidden_dim': hp.choice("hidden_dim", [4, 16, 32, 64, 128, 246])}

#Objective
def objective(params):
    config = {'model_name':"MLP"}
    for (key, value) in params.items():
        config[key] = int(value)
    #print(config)
    run = experiment.run(config_updates=config)
    print("Result:", run.result)
    return run.result

#Optimisation
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)
print("Best", best)
