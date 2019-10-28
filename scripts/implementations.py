# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import *


""" 
Implementations.py is the folder where we store all the source code for the our algorithm
for our prediction making 
"""

""" Least squares Analytical """

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(tx),tx)),np.transpose(tx)),y)
    e = y - np.matmul(tx,w)
    mse = 0.5*np.mean(e**2)
    return w, mse 
   

########################################################################################

""" Ridge regression Analytical """

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(tx),tx) + lambda_*np.eye(tx.shape[1])),np.transpose(tx)),y)
    e = y - np.matmul(tx,w)
    mse = 0.5*np.mean(e**2)
    return w, mse


####################################################################################################


"""Least Squares Gradient Descent"""

def grad_ls(y, tx, w):
    """Compute the gradient."""
    e = y - np.matmul(tx,w)
    gradient = -np.matmul(np.transpose(tx)/y.shape[0], e)
    return gradient
    
def loss_ls(y, tx, w):
    """Compute the gradient."""
    e = y - np.matmul(tx,w)
    mse = np.sum(np.multiply(e,e)/(2*y.shape[0]))
    return mse

def least_squares_GD(y, tx, initial_w, max_iters, gamma, verbose = False): #We add verbose
    """Gradient descent algorithm."""
    # Define parameters to store w and loss

    y = y.reshape(-1,1)
    w = initial_w.reshape(-1,1)
    loss = None
    for n_iter in range(max_iters):
        # compute gradient and loss, and update the weights
        grad = grad_ls(y, tx, w)
        loss = loss_ls(y, tx, w)
        w = w - gamma * grad

        if verbose: print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    return w, loss



########################################################################################

"""Least Squares Stochastic Gradient Descent"""

def least_squares_SGD(
        y, tx, initial_w, max_iters, gamma, batch_size=1,verbose = False):
    """Stochastic gradient descent algorithm."""
    y = y.reshape(-1,1)
    w = initial_w.reshape(-1,1)
    loss=None
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = grad_ls(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
        loss = loss_ls(y, tx, w)        

        if verbose: print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss


########################################################################################

"""Logistic Regression"""

def sigmoid(t):
    """apply sigmoid function on t."""
    t = np.clip(t,a_min=-40,a_max=None)
    return 1/(1+np.exp(-t))

def loss_logreg(y, tx, w):
    """compute the cost by negative log likelihood."""
    s = 0.98*sigmoid(np.matmul(tx, w))+0.01
    loss = - np.sum((y*np.log(s) + (1-y)*np.log(1-s))/y.shape[0])
    return loss

def gradient_logreg(y, tx, w):
    """compute the gradient of loss."""
    s = sigmoid(np.matmul(tx, w))
    return np.matmul(np.transpose(tx), s - y)

def grad_step_logreg(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = loss_logreg(y, tx, w)
    grad = gradient_logreg(y, tx, w)
    w = w - gamma * grad
    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma, batch_size=1,verbose =False):
    # init parameters
    threshold = 1e-8
    y = y.reshape(-1,1)
    w = initial_w.reshape(-1,1)
    loss=None

    # start the logistic regression
    for n_iter in range(max_iters):
        # get loss and update w.
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            _, w = grad_step_logreg(minibatch_y, minibatch_tx, w, gamma)
        loss = loss_logreg(y, tx, w)
        # log info
        if verbose: print("Log Regression({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss    


########################################################################################


"""Regularized Logistic Regression"""

def grad_step_reg_logreg(y, tx, lambda_, w, gamma):
    """return the loss, gradient, and hessian."""
    loss = loss_logreg(y, tx, w) + 0.5*lambda_*np.sum(w*w)
    grad = gradient_logreg(y, tx, w) + lambda_*w
    w = w - gamma * grad
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, batch_size=1, lr_decay=False, lr_decay_rate=0.7, decay_step=30,verbose = False):
    
    # init parameters
    y = y.reshape(-1,1)
    w = initial_w.reshape(-1,1)
    loss=None
    
    # start the logistic regression
    for n_iter in range(max_iters):
        # get loss and update w.
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            _, w = grad_step_reg_logreg(minibatch_y, minibatch_tx, 
                                        lambda_,w, gamma)
        lr_l=loss_logreg(y, tx, w)
        reg_l=0.5*lambda_*np.sum(w*w)
        loss = lr_l + reg_l
        # log info
        if n_iter%100==0 and verbose:
            print("Log Regression({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))

        if lr_decay and n_iter%decay_step:
                gamma = gamma*lr_decay_rate
    return w, loss



########################################################################################



def graph_reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, batch_size=1, lr_decay=False, lr_decay_rate=0.7, decay_step=30,verbose=False):
    # init parameterss
    y = y.reshape(-1,1)
    w = initial_w.reshape(-1,1)
    loss=None
    losses = []
    ws = []

    # start the logistic regression
    for n_iter in range(max_iters):

        # get loss and update w.
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            _, w = grad_step_reg_logreg(minibatch_y, minibatch_tx, 
                                        lambda_,w, gamma)
        lr_l=loss_logreg(y, tx, w)
        reg_l=0.5*lambda_*np.sum(w*w)
        loss = lr_l + reg_l

        losses.append(loss)
        ws.append(w)

        # log info
        if n_iter%100==0 and verbose:
            print("Log Regression({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))

        #Some tweaking that we did to avoid a too high learning rate after a training time
        if lr_decay and n_iter%decay_step: gamma = gamma*lr_decay_rate

    return ws,losses



