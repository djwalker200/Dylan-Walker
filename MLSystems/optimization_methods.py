#!/usr/bin/env python3
import os
import numpy as np
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot as plt

from scipy.special import softmax

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

import time

def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        numpy.random.seed(8675309)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_batch_grad(Xs, Ys, ii, gamma, W):
    
    Xs = Xs[:,ii]
    Ys = Ys[:,ii]
        
    inner = softmax(W @ Xs,axis=0) - Ys
    
    grad = inner @ Xs.T
    
    return grad / len(ii) + gamma * W


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):

    pred = np.argmax(softmax(W @ Xs,axis=0),axis=0)
    true = np.argmax(Ys,axis=0)
    
    return true[true != pred].shape[0] / Xs.shape[1]


# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
        
    (d, n) = Xs.shape
    yhat = softmax(W @ Xs,axis=0)
    l = -np.sum(Ys * np.log(yhat))
    return np.mean(l) + (gamma / 2) * np.linalg.norm(W,ord="fro")


# gradient descent (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):

    results = [W0]
    W = np.copy(W0)
    ii = np.arange(Xs.shape[1])
    for itr in range(num_epochs):
        grad = multinomial_logreg_batch_grad(Xs,Ys,ii,gamma,W)

        W = W - alpha * grad
        
        if itr % monitor_freq == 0:
            results.append(W)
           
    results.append(W)
    
    return results


# gradient descent with nesterov momentum
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gd_nesterov(Xs, Ys, gamma, W0, alpha, beta, num_epochs, monitor_period):
    
    results = [W0]
    W = np.copy(W0)
    v = np.copy(W0)
    ii = np.arange(Xs.shape[1])
    for itr in range(num_epochs):
        grad = multinomial_logreg_batch_grad(Xs,Ys,ii,gamma,W)

        vNew = W - alpha * grad
        W = vNew + beta * (vNew - v)
        v = vNew
        
        if itr % monitor_freq == 0:
            results.append(W)
           
    results.append(W)
    
    return results

# SGD: run stochastic gradient descent with minibatching and sequential sampling order (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    
    d,n = Xs.shape
    results = [W0]
    W = np.copy(W0)
    batch_count = 0

    for epoch in range(num_epochs):
        
        
        for start_idx in range(0,n,B):
            
            ii = np.arange(start_idx,start_idx + B)
            grad = multinomial_logreg_batch_grad(Xs,Ys,ii,gamma,W)

            W = W - alpha * grad
        
            if batch_count % monitor_period == 0:
                results.append(W)
                
            batch_count += 1
            
    results.append(W)
    return results

# SGD + Momentum: add momentum to the previous algorithm
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, monitor_period):
    
    d,n = Xs.shape
    results = [W0]
    W = np.copy(W0)
    v = 0
    batch_count = 0

    for epoch in range(num_epochs):
        
        
        for start_idx in range(0,n,B):
            
            ii = np.arange(start_idx,start_idx + B)
            grad = multinomial_logreg_batch_grad(Xs,Ys,ii,gamma,W)

            v = beta * v - alpha * grad
            W = W + v
            
            if batch_count % monitor_period == 0:
                results.append(W)
                
            batch_count += 1
    
    results.append(W)
    return results

# Adam Optimizer
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# rho1            first moment decay rate ρ1
# rho2            second moment decay rate ρ2
# B               minibatch size
# eps             small factor used to prevent division by zero in update step
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def adam(Xs, Ys, gamma, W0, alpha, rho1, rho2, B, eps, num_epochs, monitor_period):

    d,n = Xs.shape
    results = [W0]
    W = np.copy(W0)
    
    r = np.zeros(W.shape)
    s = np.zeros(W.shape)
    
    batch_count = 0
    t = 0
    for epoch in range(num_epochs):
        t = t + 1
        
        for start_idx in range(0,n,B):
            
            ii = np.arange(start_idx,start_idx + B)
            grad = multinomial_logreg_batch_grad(Xs,Ys,ii,gamma,W)

            s = rho1 * s + (1 - rho1) * grad
            r = rho2 * r + (1 - rho2) * np.square(grad)
            
            sHat = s / (1 - np.power(rho1,t))
            rHat = r / (1 - np.power(rho2,t))
            
            W = W - alpha * (sHat / np.sqrt(rHat + eps)) 
        
            if batch_count % monitor_period == 0:
                results.append(W)
                
            batch_count += 1

    results.append(W)
    return results


def plotLossesErrors(Xs,Ys,gamma,Ws,name,train=True):
    errors = []
    losses = []

    for w in Ws:
        errors.append(multinomial_logreg_error(Xs, Ys, w))
        losses.append(multinomial_logreg_loss(Xs,Ys,gamma,w))

        
    if train:
        labels = ['Train Error','Train Losses']
        name += "_train"
    else:
        labels = ['Test Error','Test Losses']
        name += "_test"
        
    
    plt.plot(errors,label=labels[0])
    plt.legend()
    plt.title(labels[0])
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.savefig(name + "_errors.jpg")
    plt.clf()
    
    if not train:
        return
    
    plt.plot(losses,label=labels[1])
    plt.legend()
    plt.title(labels[1])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(name + "_losses.jpg")
    plt.clf()
    
if __name__ == "__main__":
    
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()

    
    d,n = Xs_tr.shape
    c,_ = Ys_tr.shape
    W0 = random.randn(c,d)
    
    
    
    # Hyperparameters
    gamma = 0.0001
    alpha = 1.0
    epochs = 100
    monitor_freq = 1
    
    # Gradient Descent
    start = time.time()
    gd = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, epochs, monitor_freq)
    print(f'Finished Gradient Descent in {time.time() - start} seconds')
    
    
    # Nesterov Momentum
    beta = 0.9
    start = time.time()
    nesterov1 = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta, epochs, monitor_freq)
    print(f"Finished Nesterov in {time.time() - start} seconds")
    

    
    # Plot train and test for each model
    plotLossesErrors(Xs_tr,Ys_tr,gamma,gd,"Gradient Descent",train=True)
    plotLossesErrors(Xs_te,Ys_te,gamma,gd,"Gradient Descent",train=False)

    plotLossesErrors(Xs_tr,Ys_tr,gamma,nesterov1,"Nesterov",train=True)
    plotLossesErrors(Xs_te,Ys_te,gamma,nesterov1,"Nesterov",train=False)
    
    
    # Reset Hyperparameters
    gamma = 0.0001
    alpha = 0.2
    epochs = 10
    B = 600
    monitor_freq = 10
    
    # SGD
    start = time.time()
    sgd = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B,epochs, monitor_freq)
    print(f'Finished SGD in {time.time() - start} seconds')

    
    # SGD with Momentum
    beta = 0.9
    start = time.time()
    sgd_momentum = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, epochs, monitor_freq)
    print(f'Finished SGD Momentum 1 in {time.time() - start} seconds')

    
    # Plot train and test for SGD models
    plotLossesErrors(Xs_tr,Ys_tr,gamma,sgd,"SGD",train=True)
    plotLossesErrors(Xs_te,Ys_te,gamma,sgd,"SGD",train=False)
    
    plotLossesErrors(Xs_tr,Ys_tr,gamma,sgd_momentum,"SGD w/ Momentum",train=True)
    plotLossesErrors(Xs_te,Ys_te,gamma,sgd_momentum,"SGD w/ Momentum",train=False)

    
    
    # Reset Hyperparameters
    gamma = 0.0001
    alpha = 0.01
    epochs = 10
    B = 600
    monitor_freq = 10

    
    # ADAM Hyperparameters
    rho1 = 0.9
    rho2 = 0.999
    eps = 1e-5
    
    # ADAM
    start = time.time()
    adam_results = adam(Xs_tr, Ys_tr, gamma, W0, alpha, rho1, rho2, B, eps, epochs, monitor_freq)
    print(f'Finished ADAM in {time.time() - start} seconds')

    
    # Plot train and test for ADAM
    plotLossesErrors(Xs_tr,Ys_tr,gamma,adam_results,"ADAM",train=True)
    plotLossesErrors(Xs_te,Ys_te,gamma,adam_results,"ADAM",train=False)
    