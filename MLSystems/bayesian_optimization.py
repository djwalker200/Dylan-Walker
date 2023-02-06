#!/usr/bin/env python3
import os
import math
import matplotlib
import pickle
import numpy
import time
import scipy.special
import mnist
from tqdm import tqdm
matplotlib.use('agg')
from matplotlib import pyplot
from matplotlib import animation
import torch


mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")


def load_MNIST_dataset_with_validation_split():
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
        # extract out a validation set
        Xs_va = Xs_tr[:,50000:60000]
        Ys_va = Ys_tr[:,50000:60000]
        Xs_tr = Xs_tr[:,0:50000]
        Ys_tr = Ys_tr[:,0:50000]
        # load test data
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_va, Ys_va, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# computes the cumulative distribution function of a standard Gaussian random variable
def gaussian_cdf(u):
    return 0.5*(1.0 + torch.special.erf(u/math.sqrt(2.0)))

# computes the probability mass function of a standard Gaussian random variable
def gaussian_pmf(u):
    return torch.exp(-u**2/2.0)/math.sqrt(2.0*math.pi)


# computes the Gaussian RBF kernel matrix for a vector of data points 
#
# Xs        points at which to compute the kernel (size: d x m)
# Zs        other points at which to compute the kernel (size: d x n)
# gamma     gamma parameter for the RBF kernel
#
# returns   an (m x n) matrix Sigma where Sigma[i,j] = K(Xs[:,i], Zs[:,j])
def rbf_kernel_matrix(Xs, Zs, gamma):

    norms = torch.norm(torch.unsqueeze(Xs, -1) - torch.unsqueeze(Zs, 1),dim=0)
    return torch.exp(-gamma * torch.square(norms))
    

# computes the distribution predicted by a Gaussian process that uses an RBF kernel (in PyTorch)
#
# Xs            points at which to compute the kernel (size: d x n) where d is the number of parameters
# Ys            observed value at those points (size: n)
# gamma         gamma parameter for the RBF kernel
# sigma2_noise  the variance sigma^2 of the additive gaussian noise used in the model
#
# returns   a function that takes a value Xtest (size: d) and returns a tuple (mean, variance)
def gp_prediction(Xs, Ys, gamma, sigma2_noise):
    
    d, n = Xs.shape

    K = rbf_kernel_matrix(Xs, Xs, gamma) # n x n
    sigma_inv = torch.inverse(K + sigma2_noise * torch.eye(n))
    Ky = sigma_inv @ Ys # n x 1
    
    
    def prediction_mean_and_variance(Xtest):
        
        Xtest = Xtest.reshape(-1,1)
        K_star = rbf_kernel_matrix(Xs, Xtest, gamma).type(torch.FloatTensor) # n x 1
        K_star_star = rbf_kernel_matrix(Xtest, Xtest, gamma) # 1 x 1
        
        mean = K_star.T @ Ky
        variance = K_star_star + sigma2_noise - K_star.T @ sigma_inv @ K_star
        return (mean.reshape(()), variance.reshape(())) 
    

    return prediction_mean_and_variance


# computes the probability of improvement (PI) acquisition function
#
# Ybest     value at best "y"
# mean      mean of prediction
# stdev     standard deviation of prediction (the square root of the variance)
#
# returns   PI acquisition function
def pi_acquisition(Ybest, mean, stdev):

    inner = (Ybest - mean) / stdev
    pi = -gaussian_cdf(inner)
    return pi


# computes the expected improvement (EI) acquisition function
#
# Ybest     value at best "y"
# mean      mean of prediction
# stdev     standard deviation of prediction
#
# returns   EI acquisition function
def ei_acquisition(Ybest, mean, stdev):
    
    inner = (Ybest - mean) / stdev
    result = gaussian_pmf(inner) + inner * gaussian_cdf(inner)
    return -stdev * result

# returns a function that computes the lower confidence bound (LCB) acquisition function
#
# kappa     parameter for LCB
#
# returns   function that computes the LCB acquisition function
def lcb_acquisition(kappa):
    
    def A_lcb(Ybest, mean, stdev):
        
        return mean - kappa * stdev
    
    return A_lcb


# gradient descent to do the inner optimization step of Bayesian optimization
#
# objective     the objective function to minimize, as a function that takes a torch tensor and returns an expression
# x0            initial value to assign to variable (torch tensor)
# alpha         learning rate/step size
# num_iters     number of iterations of gradient descent
#
# returns     (obj_min, x_min), where
#       obj_min     the value of the objective after running iterations of gradient descent
#       x_min       the value of x after running iterations of gradient descent
def gradient_descent(objective, x0, alpha, num_iters):
    
    x = x0.detach().clone()  
    x.requires_grad = True  
    opt = torch.optim.SGD([x], alpha)
    
    for it in range(num_iters):
        
        opt.zero_grad()
        f = objective(x)
        f.backward()
        opt.step()
        
    x.requires_grad = False 
    
    return (float(f.item()), x)

# runs Bayesian optimization to minimize an objective
#
# objective     objective function; takes a torch tensor, returns a python float scalar
# d             dimension to optimize over
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# acquisition   acquisition function to use (e.g. ei_acquisition)
# random_x      function that returns a random sample of the parameter we're optimizing over (a torch tensor, e.g. for use in warmup)
# gd_nruns      number of random initializations we should use for gradient descent for the inner optimization step
# gd_alpha      learning rate for gradient descent
# gd_niters     number of iterations for gradient descent
# n_warmup      number of initial warmup evaluations of the objective to use
# num_iters     number of outer iterations of Bayes optimization to run (including warmup)
#
# returns       tuple of (y_best, x_best, Ys, Xs), where
#   y_best          objective value of best point found
#   x_best          best point found
#   Ys              vector of objective values for all points searched (size: num_iters)
#   Xs              matrix of all points searched (size: d x num_iters)
def bayes_opt(objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters):

    y_best = None
    x_best = None
    Xs = torch.zeros((d,num_iters))
    Ys = torch.zeros((num_iters,))
    
    time_sum = 0
    
    for i in range(n_warmup):
    
        x = random_x()
        
        start = time.time()
        y = objective(x)     
        time_sum += time.time() - start
        
        # Store and update values
        Xs[:,i] = x
        Ys[i] = y
        if y_best == None or y <= y_best:
            y_best = y
            x_best = x
    
    
    for i in range(n_warmup,num_iters):
        
        
        # Function for mean and variance
        pred_mean_and_variance = gp_prediction(Xs, Ys, gamma, sigma2_noise)
        
        # Target function for gradient descent to minimize acquisition
        def x_acquisition(x):
            
            (mean,var) = pred_mean_and_variance(x)
            val = acquisition(y_best,mean,torch.sqrt(var))
            return val
        
        # Stores best values from gradient descent
        lowest_val = None
        x = None
        
        # Run gradient descent with gd_nruns random initializations
        for j in range(gd_nruns):
            
            x0 = random_x()

            (val,xT) = gradient_descent(x_acquisition, x0, gd_alpha, gd_niters)
            
            # Check if best run
            if lowest_val is None or val < lowest_val:
                lowest_val = val
                x = xT
    
        
        # Compute objective for best x
        start = time.time()
        y = objective(x)
        time_sum += time.time() - start  
        if torch.is_tensor(y):
            y = y.item()
        
        # Store and update values
        Xs[:,i] = x
        Ys[i] = y
        if y <= y_best:
            y_best = y
            x_best = x
    
    return (y_best,x_best,Ys,Xs)

# a one-dimensional test objective function on which to run Bayesian optimization
def test_objective(x):
    
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1,)
    x = x.item() 
    return (math.cos(8.0*x) - 0.3 + (x-0.5)**2)



# computes the gradient of the multinomial logistic regression objective with regularization 
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_batch_grad(Xs, Ys, ii, gamma, W):
    
    inner = scipy.special.softmax(W @ Xs[:,ii], axis=0) - Ys[:,ii]
    return (inner @ Xs[:,ii].T / len(ii)) + gamma * W


# computes the error of the classifier 
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):

    predictions = numpy.argmax(numpy.dot(W, Xs), axis=0)
    error = numpy.mean(predictions != numpy.argmax(Ys, axis=0))
    return error


# computes the cross-entropy loss of the classifier 
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):

    (d, n) = Xs.shape
    return -numpy.sum(numpy.log(scipy.special.softmax(numpy.dot(W, Xs), axis=0)) * Ys) / n + (gamma / 2) * (numpy.linalg.norm(W, "fro")**2)


# SGD + Momentum: add momentum to the previous algorithm
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs 
#
# returns         the final model, after training
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):

    (d, n) = Xs.shape
    V = numpy.zeros(W0.shape)
    W = W0
    niter = 0
    print("Running minibatch sequential-scan SGD with momentum")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            niter += 1
            ii = range(ibatch*B, (ibatch+1)*B)
            V = beta * V - alpha * multinomial_logreg_batch_grad(Xs, Ys, ii, gamma, W)
            W = W + V

    return W


# produce a function that runs SGD+Momentum on the MNIST dataset, initializing the weights to zero
#
# mnist_dataset         the MNIST dataset, as returned by load_MNIST_dataset_with_validation_split
# num_epochs            number of epochs
# B                     the batch size
#
# returns               a function that takes parameters
#   params                  a numpy vector of shape (3,) with entries that determine the hyperparameters, where
#       gamma = 10^(-8 * params[0])
#       alpha = 0.5*params[1]
#       beta = params[2]
#                       and returns (the validation error of the final trained model after all the epochs) minus 0.9.
#                       if training diverged (i.e. any of the weights are non-finite) then returns 0.1, which corresponds to an error of 1.
def mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B):
    # TODO students should implement this

    (Xs_tr, Ys_tr, Xs_va, Ys_va, __, __) = mnist_dataset
    
    (d,n) = Xs_tr.shape
    (c,n) = Ys_tr.shape
    
    W0 = numpy.zeros((c,d))
    
    def mnist_sgd(params):
        
        gamma = 10 ** (-8 * params[0].item())
        alpha = 0.5 * params[1].item()
        beta = params[2].item()
        W = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
        
        if numpy.isnan(W).any():
            return 0.1
        
        else:
            val_error = multinomial_logreg_error(Xs_va, Ys_va, W)
            return val_error - 0.9
        
        
    return mnist_sgd


if __name__ == "__main__":

    
    # Small test example
    def test_random_x():
        return 1.5 * torch.rand(1) - 0.25
    
    
    def rand_uniform_x():
        return torch.rand(1)
    
    
    
    # Hyperparameters
    gamma = 10.0
    var = 0.001
    lr = 0.01
    n_warmup = 3
    niters = 20
    gd_runs = 20
    gd_steps = 20
    
    
    # Tests all 3 acquisition functions on the simple test function
    
    (y_best_ei, x_best_ei, Ys_ei, Xs_ei) = bayes_opt(test_objective, 1, gamma, var,
                                         ei_acquisition, rand_uniform_x, gd_runs, lr, gd_steps, n_warmup,niters)
    
    print(f"Best parameter value for EI acquisition: {x_best_ei} with function value {y_best_ei}")
    
    
    (y_best_pi, x_best_pi, Ys_pi, Xs_pi) = bayes_opt(test_objective, 1, gamma, var,
                                         pi_acquisition, rand_uniform_x, gd_runs, lr, gd_steps, n_warmup,niters)
    
    print(f"Best parameter value for PI acquisition: {x_best_pi} with function value {y_best_pi}")
    
    
    kappa = 2
    lcb = lcb_acquisition(kappa)
    
    (y_best_lcb, x_best_lcb, Ys_lcb, Xs_lcb) = bayes_opt(test_objective, 1, gamma, var,
                                         lcb, rand_uniform_x, gd_runs, lr, gd_steps, n_warmup,niters)
    
    print(f"Best parameter value for LCB acquisition: {x_best_lcb} with function value {y_best_lcb}")
    
    
    
    
    # Runs Bayesian Optimization on MNIST over gamma (regularization parameter), alpha (learning rate), and beta (momentum parameter)
    
    B = 500
    epochs = 5
    
    kappa = 2
    lcb = lcb_acquisition(kappa)
    
    
    mnist_dataset = load_MNIST_dataset_with_validation_split()
    mnist_sgd = mnist_sgd_mss_with_momentum(mnist_dataset,epochs, B)
    
    
    def rand_uniform_x3():
        return torch.rand(3)
    
    
    start = time.time()
    (y_best_sgd, x_best_sgd, Ys_sgd, Xs_sgd) = bayes_opt(mnist_sgd, 3, gamma, var,
                                         lcb, rand_uniform_x3, gd_runs, lr, gd_steps, n_warmup,niters)
    
    print(f"Total runtime of bayesian optimization: {time.time() - start}")
    
    print(f"Best parameter value for lcb acquisition with kappa = {kappa}: {x_best_sgd} with function value {y_best_sgd}")
    
    
    (Xs_tr, Ys_tr, Xs_va, Ys_va, Xs_te, Ys_te) = mnist_dataset
    
    (d,n) = Xs_tr.shape
    (c,n) = Ys_tr.shape
    
    W0 = numpy.zeros((c,d))
    
    gamma = 10 ** (-8 * x_best_sgd[0].item())
    alpha = 0.5 * x_best_sgd[1].item()
    beta = x_best_sgd[2].item()
    
    W = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, epochs)
    val_error = multinomial_logreg_error(Xs_va, Ys_va, W)
    test_error = multinomial_logreg_error(Xs_te, Ys_te, W)
    
    print(f"Final Validation Error: {val_error}")
    print(f"Final Test Error: {test_error}")
    
    '''