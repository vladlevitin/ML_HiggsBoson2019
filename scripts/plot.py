import data_processor as dp
from implementations import *
import numpy as np
import matplotlib.pyplot as plt

def plot_logistic(losses,ws,tX_val,y_val):
    # Plot two plots:
    # loss as a function of step
    # accuracy as a function of step  (on validation set)

    steps = [i for i in range(1,len(losses)+1)]

    #plotting first graph
    ax1 = plt.plot(steps,losses)
    plt.title('Loss as a function of step')
    plt.xlabel('number of steps')
    plt.ylabel('loss (approximate)')
    plt.show()

    #compute accuracy
    accuracy = []
    for w in ws:
        y_pred = predict_labels(w,tX_val)
        accuracy.append(np.mean(y_val.reshape(-1,1)==y_pred))

    #plotting second graph 
    ax2 = plt.plot(steps,accuracy)
    plt.title('Accuracy as a function of step')
    plt.xlabel('number of steps')
    plt.ylabel('accuracy on validation set')
    plt.show()

def plot_ridge(lambdas, accuracies):

    plt.scatter(lambdas,accuracies)
    plt.title('Accuracy as a function of the lambda used')
    plt.xlabel('different lambdas used')
    plt.ylabel('accuracy given for each lambda')
    plt.show()