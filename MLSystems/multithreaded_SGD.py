#!/usr/bin/env python3
import os

# Thread settings for implicit parallelization
implicit_num_threads = 1
os.environ["OMP_NUM_THREADS"] = str(implicit_num_threads)
os.environ["MKL_NUM_THREADS"] = str(implicit_num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(implicit_num_threads)
# END THREAD SETTINGS

import numpy as np
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot as plt
import threading
import time

from tqdm import tqdm

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")


# Helper function that computes logistic regression error for a given set of weights 
def multinomial_logreg_error(Xs, Ys, W):
    
    predictions = np.argmax(np.dot(W, Xs), axis=0)
    error = np.mean(predictions != np.argmax(Ys, axis=0))
    return error

# Helper function that computes logistic regression gradient for a given batch
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    
    WdotX = np.dot(W, Xs[:,ii])
    expWdotX = np.exp(WdotX - np.amax(WdotX, axis=0))
    softmaxWdotX = expWdotX / np.sum(expWdotX, axis = 0)
    
    return np.dot(softmaxWdotX - Ys[:,ii], Xs[:,ii].transpose()) / len(ii) + gamma * W



def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="np", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        np.random.seed(4787)
        perm = np.random.permutation(60000)
        Xs_tr = np.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = np.ascontiguousarray(Ys_tr[:,perm])
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset



# SGD + Momentum 
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              initial parameters (c * d)
# alpha           learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):

    models = []
    (d, n) = Xs.shape
    V = np.zeros(W0.shape)
    W = W0
    print("Running minibatch sequential-scan SGD with momentum")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            ii = range(ibatch*B, (ibatch+1)*B)
            V = beta * V - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            W = W + V
            
    return W


# SGD + Momentum with all memory preallocated before inner loop
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              initial parameters (c * d)
# alpha           learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs 
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_noalloc(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    
    (d, n) = Xs.shape
    (c, d) = W0.shape

    W = W0
    V = np.zeros(W0.shape)
    
    grad = np.zeros(W0.shape)
    decay = np.zeros(W0.shape)
    
    WdotX = np.zeros((c,B))
    temp = np.zeros((B,))
    
    Xs_cont = []
    Ys_cont = []
    for batch in range(int(n/B)):
        ii = range(batch*B, (batch+1)*B)
        Xs_cont.append(np.ascontiguousarray(Xs[:,ii]))
        Ys_cont.append(np.ascontiguousarray(Ys[:,ii]))
    
    print("Running minibatch sequential-scan SGD with momentum (no allocation)")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):

            X_slice = Xs_cont[ibatch]
            Y_slice = Ys_cont[ibatch]
            
            # WdotX = W * Xs[:,ii]
            np.dot(W, X_slice,out=WdotX)
            # Amax = amax(WdotX,axis=0)
            np.amax(WdotX,axis=0,out=temp)
            #  WdotX - Amax
            np.subtract(WdotX,temp,out=WdotX)
            # exp(WdotX - Amax)
            np.exp(WdotX,out=WdotX)
            
            #expSum = sum(exp(WdotX - Amax), axis=0)
            np.sum(WdotX,axis=0,out=temp)
            # sum(exp(WdotX - Amax, axis=0) / expSum
            np.divide(WdotX,temp,out=WdotX)
            
            # sum(exp(WdotX - Amax, axis=0) / expSum - Ys[:,ii]
            np.subtract(WdotX,Y_slice,out=WdotX)
            XT = X_slice.transpose()
            
            # grad = (sum(exp(WdotX - Amax, axis=0) / expSum - Ys[:,ii]) @ Xs[:,ii].T
            np.dot(WdotX,XT,out=grad)
        
            # grad / B
            np.divide(grad,B,out=grad)
            # decay = gamma * W
            np.multiply(gamma,W,out=decay)
            # grad + decay
            np.add(grad,decay,out=grad)
            
            
                  
            # alpha * grad
            np.multiply(alpha,grad,out=grad)
            # beta * V
            np.multiply(beta,V,out=V)
            # V = beta * V - alpha * grad
            np.subtract(V,grad,out=V)
            # W = W + V
            np.add(W,V,out=W)
            
    return W


# SGD + Momentum (threaded)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              initial parameters (c * d)
# alpha           learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_threaded(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    
    (d, n) = Xs.shape
    (c, d) = W0.shape

    W = W0
    V = np.zeros(W0.shape)
    grad = np.zeros(W0.shape)
    
    Bprime = B // num_threads
    all_grads = np.zeros((num_threads,c,d))
    
    Xs_cont = []
    Ys_cont = []
    for batch in range(int(n/B)):
        ii = range(batch*B, (batch+1)*B)
        Xs_cont.append(np.ascontiguousarray(Xs[:,ii]))
        Ys_cont.append(np.ascontiguousarray(Ys[:,ii]))
    
    
    # Constructs the threading barrier object
    iter_barrier = threading.Barrier(num_threads + 1)

    # Function for each thread to run
    def thread_main(ithread):
        
        grad = np.zeros(W0.shape)
        decay = np.zeros(W0.shape)

        WdotX = np.zeros((c,Bprime))
        temp = np.zeros((Bprime,))
    
        for it in range(num_epochs):
            for ibatch in range(int(n/B)):
                
                # Gets slice indices
                ii = range(ithread * Bprime,(ithread + 1) * Bprime)
                
                
                # Grab parallel minibatch
                X_slice = Xs_cont[ibatch][:,ii]
                Y_slice = Ys_cont[ibatch][:,ii]

                # WdotX = W * Xs[:,ii]
                np.dot(W, X_slice,out=WdotX)
                # Amax = amax(WdotX,axis=0)
                np.amax(WdotX,axis=0,out=temp)
                #  WdotX - Amax
                np.subtract(WdotX,temp,out=WdotX)
                # exp(WdotX - Amax)
                np.exp(WdotX,out=WdotX)

                #expSum = sum(exp(WdotX - Amax), axis=0)
                np.sum(WdotX,axis=0,out=temp)
                # sum(exp(WdotX - Amax, axis=0) / expSum
                np.divide(WdotX,temp,out=WdotX)

                # sum(exp(WdotX - Amax, axis=0) / expSum - Ys[:,ii]
                np.subtract(WdotX,Y_slice,out=WdotX)
                XT = X_slice.transpose()

                # grad = (sum(exp(WdotX - Amax, axis=0) / expSum - Ys[:,ii]) @ Xs[:,ii].T
                np.dot(WdotX,XT,out=grad)

                # grad / B
                np.divide(grad,B,out=grad)
                # decay = gamma * W
                np.multiply(gamma,W,out=decay)
                # grad + decay
                np.add(grad,decay,out=all_grads[ithread,:,:])
                
                # Wait at barrier for all threads to finish
                iter_barrier.wait()
                
                # Allow for computation of full gradient
                iter_barrier.wait()
                

    # Creates worker threads
    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]

    
    # Initializes all workers
    for t in worker_threads:
        print("running thread ", t)
        t.start()

        
    print("Running minibatch sequential-scan SGD with momentum (%d threads)" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            
            # Run gradient computation on all threads before first barrier
            
            iter_barrier.wait()    
                
            # Compute the full gradient across all threads
            
            # Sum over each of the thread gradients
            np.sum(all_grads,axis=0,out=grad)
            
            # alpha * grad
            np.multiply(alpha,grad,out=grad)
            # beta * V
            np.multiply(beta,V,out=V)
            # V = beta * V - alpha * grad
            np.subtract(V,grad,out=V)
            # W = W + V
            np.add(W,V,out=W)
            
            
            # Wait on barrier before starting next iteration
            iter_barrier.wait()

            

    for t in worker_threads:
        t.join()
        
    # returns the final model
    return W


# SGD + Momentum with preallocation and 32-bit arithmetic
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              initial parameters (c * d)
# alpha           learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs 
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_noalloc_float32(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    
    (d, n) = Xs.shape
    (c, d) = W0.shape
    
    W = W0.astype(np.float32)
    V = np.zeros(W0.shape,dtype=np.float32)
    
    grad = np.zeros(W0.shape,dtype=np.float32)
    decay = np.zeros(W0.shape,dtype=np.float32)
    
    WdotX = np.zeros((c,B),dtype=np.float32)
    temp = np.zeros((B,),dtype=np.float32)
    
    Xs = Xs.astype(np.float32)
    Ys = Ys.astype(np.float32)
    
    Xs_cont = []
    Ys_cont = []
    for batch in range(int(n/B)):
        ii = range(batch*B, (batch+1)*B)
        Xs_cont.append(np.ascontiguousarray(Xs[:,ii]))
        Ys_cont.append(np.ascontiguousarray(Ys[:,ii]))
    
    print("Running minibatch sequential-scan SGD with momentum (no allocation)")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):

            X_slice = Xs_cont[ibatch]
            Y_slice = Ys_cont[ibatch]
            
            
            # WdotX = W * Xs[:,ii]
            np.dot(W, X_slice,out=WdotX)
            # Amax = amax(WdotX,axis=0)
            np.amax(WdotX,axis=0,out=temp)
            #  WdotX - Amax
            np.subtract(WdotX,temp,out=WdotX)
            # exp(WdotX - Amax)
            np.exp(WdotX,out=WdotX)
            
            #expSum = sum(exp(WdotX - Amax), axis=0)
            np.sum(WdotX,axis=0,out=temp)
            # sum(exp(WdotX - Amax, axis=0) / expSum
            np.divide(WdotX,temp,out=WdotX)
            
            # sum(exp(WdotX - Amax, axis=0) / expSum - Ys[:,ii]
            np.subtract(WdotX,Y_slice,out=WdotX)
            XT = X_slice.transpose()
            
            # grad = (sum(exp(WdotX - Amax, axis=0) / expSum - Ys[:,ii]) @ Xs[:,ii].T
            np.dot(WdotX,XT,out=grad)
        
            # grad / len(ii)
            np.divide(grad,B,out=grad)
            # decay = gamma * W
            np.multiply(gamma,W,out=decay)
            # grad + decay
            np.add(grad,decay,out=grad)
            
                  
            # alpha * grad
            np.multiply(alpha,grad,out=grad)
            # beta * V
            np.multiply(beta,V,out=V)
            # V = beta * V - alpha * grad
            np.subtract(V,grad,out=V)
            # W = W + V
            np.add(W,V,out=W)

            
    return W
    


# SGD + Momentum (threaded) with 32-bit arithmetic
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              initial parameters (c * d)
# alpha           learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_threaded_float32(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    
    (d, n) = Xs.shape
    (c, d) = W0.shape

    W = W0.astype(np.float32)
    V = np.zeros(W0.shape,dtype=np.float32)
    main_grad = np.zeros(W0.shape,dtype=np.float32)
    
    Bprime = B // num_threads
    all_grads = np.zeros((num_threads,c,d),dtype=np.float32)
    
    Xs = Xs.astype(np.float32)
    Ys = Ys.astype(np.float32)
    
    Xs_cont = []
    Ys_cont = []
    for batch in range(int(n/B)):
        ii = range(batch*B, (batch+1)*B)
        Xs_cont.append(np.ascontiguousarray(Xs[:,ii]))
        Ys_cont.append(np.ascontiguousarray(Ys[:,ii]))
    
    # Constructs the threading barrier object
    iter_barrier = threading.Barrier(num_threads + 1)

    # Function for each thread to run
    def thread_main(ithread):
        
        grad = np.zeros(W0.shape,dtype=np.float32)
        decay = np.zeros(W0.shape,dtype=np.float32)

        WdotX = np.zeros((c,Bprime),dtype=np.float32)
        temp = np.zeros((Bprime,),dtype=np.float32)
    
        for it in range(num_epochs):
            for ibatch in range(int(n/B)):
                
                # Collects batch indices and slices batch
                ii = range(ithread * Bprime,(ithread + 1) * Bprime) 
                
                # Grab parallel minibatch
                X_slice = Xs_cont[ibatch][:,ii]
                Y_slice = Ys_cont[ibatch][:,ii]

                # WdotX = W * Xs[:,ii]
                np.dot(W, X_slice,out=WdotX)
                # Amax = amax(WdotX,axis=0)
                np.amax(WdotX,axis=0,out=temp)
                #  WdotX - Amax
                np.subtract(WdotX,temp,out=WdotX)
                # exp(WdotX - Amax)
                np.exp(WdotX,out=WdotX)

                #expSum = sum(exp(WdotX - Amax), axis=0)
                np.sum(WdotX,axis=0,out=temp)
                # sum(exp(WdotX - Amax, axis=0) / expSum
                np.divide(WdotX,temp,out=WdotX)

                # sum(exp(WdotX - Amax, axis=0) / expSum - Ys[:,ii]
                np.subtract(WdotX,Y_slice,out=WdotX)
                XT = X_slice.transpose()

                # grad = (sum(exp(WdotX - Amax, axis=0) / expSum - Ys[:,ii]) @ Xs[:,ii].T
                np.dot(WdotX,XT,out=grad)

                # grad / B
                np.divide(grad,B,out=grad)
                # decay = gamma * W
                np.multiply(gamma,W,out=decay)
                # grad + decay
                np.add(grad,decay,out=all_grads[ithread,:,:])
                
                
                # Wait on barrier for all threads to finish
                iter_barrier.wait()
                
                # Compute full gradient before starting next iteration
                iter_barrier.wait()
                

    # Initialize worker threads
    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]
    for t in worker_threads:
        print("running thread ", t)
        t.start()

        
    
    print("Running minibatch sequential-scan SGD with momentum (%d threads)" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            
            # Wait for all threads to compute gradient estimate
            iter_barrier.wait()
            
            # Compute full gradient across all threads
                
            # Sum over each of the thread gradients
            np.sum(all_grads,axis=0,out=main_grad)
            
            # alpha * grad
            np.multiply(alpha,main_grad,out=main_grad)
            # beta * V
            np.multiply(beta,V,out=V)
            # V = beta * V - alpha * grad
            np.subtract(V,main_grad,out=V)
            # W = W + V
            np.add(W,V,out=W)
            
            
            # Start next iteration
            iter_barrier.wait()

    for t in worker_threads:
        t.join()

    # Return the learned model
    return W



if __name__ == "__main__":
    
    # Loads data
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()

    
    # Hyperparameters
    (d,n) = Xs_tr.shape
    (c,n) = Ys_tr.shape
    W0 = np.random.rand(c,d)
    gamma = 0.0001
    alpha = 0.1
    beta = 0.9
    num_epochs = 20
    batch_sizes = [8,16,30,60,200,600,3000]
    
    
    
    
    # Compares SGD runtime on standard SGD versus preallocated method
    single_64 = []
    single_noalloc_64 = []
    
    for B in batch_sizes:
    
        print(f"Computing for batch size {B}")
        
        start = time.time()
        W1 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
        #print(f"Clock time: {time.time() - start}")
        single_64.append(time.time() - start)


        start = time.time()
        W2 = sgd_mss_with_momentum_noalloc(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
        #print(f"Clock time: {time.time() - start}")
        single_noalloc_64.append(time.time() - start)

        print(f"Check: {np.isclose(W1,W2).all()}")

    print(f"Single Thread 64 bit standard alloc times: {single_64}")
    print(f"Single Thread 64 bit noalloc times: {single_noalloc_64}")
       
        

    # Runs SGD with explicit multithreading
    
    explicit_64 = []
    for B in batch_sizes:
    
        print(f"Computing for batch size {B}")

        start = time.time()
        W2 = sgd_mss_with_momentum_threaded(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs,4)
        #print(f"Clock time: {time.time() - start}")
        explicit_64.append(time.time() - start)
    
    
    print(f"Explicit 64 bit times: {explicit_64}")
    
    
    
    # Runs SGD with 32 bit arithmetic for both preallocation and explicit threading methods
    noalloc_32 = []
    threaded_32 = []
                   
    for B in batch_sizes:
    
        print(f"Computing for batch size {B}")
        
        start = time.time()
        W1 = sgd_mss_with_momentum_noalloc_float32(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
        #print(f"Clock time: {time.time() - start}")
        noalloc_32.append(time.time() - start)
        

        start = time.time()
        W2 = sgd_mss_with_momentum_threaded_float32(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs,4)
        #print(f"Clock time: {time.time() - start}")
        threaded_32.append(time.time() - start)
    

    print(f"Single Thread 32 bit noalloc times: {noalloc_32}")
    print(f"Explicit Multithread 32 bit standard alloc times: {threaded_32}")
    
    
    


    
