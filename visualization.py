import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import click

@click.command()
@click.option("--id", default=10)

def plot_graph(id):
    #Load data
    id = str(id)
    train = pd.read_csv('results/'+id+'/train_metrics.csv')
    val =  pd.read_csv('results/'+id+'/val_metrics.csv')

    #Plot loss
    sns.set(style="darkgrid")
    sns.lineplot(x="epoch", y="loss", data=train, label="Train")
    sns.lineplot(x="epoch", y="loss", data=val, label="Validation")
    plt.show()

    sns.set(style="darkgrid")
    sns.lineplot(x="epoch", y="accuracy", data=train, label="Train")
    sns.lineplot(x="epoch", y="accuracy", data=val, label="Validation")
    plt.show()

if __name__ == '__main__':
    plot_graph()
