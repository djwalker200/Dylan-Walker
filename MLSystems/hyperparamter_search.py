#!/usr/bin/env python3
import os
import time
import numpy as np
import scipy
import matplotlib
import mnist
import pickle

matplotlib.use("agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision



# load the MNIST dataset using TensorFlow/Keras
def load_MNIST_dataset():
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )
    return (train_dataset, test_dataset)


# construct dataloaders for the MNIST dataset
#
# train_dataset        input train dataset (output of load_MNIST_dataset)
# test_dataset         input test dataset (output of load_MNIST_dataset)
# batch_size           batch size for training
# shuffle_train        boolean: whether to shuffle the training dataset
#
# returns              tuple of (train_dataloader, test_dataloader)
#     each component of the tuple should be a torch.utils.data.DataLoader object
#     for the corresponding training set;
#     use the specified batch_size and shuffle_train values for the training DataLoader;
#     use a batch size of 100 and no shuffling for the test data loader
def construct_dataloaders(train_dataset, test_dataset, batch_size, shuffle_train=True):

    return (
        DataLoader(train_dataset, batch_size, shuffle_train),
        DataLoader(test_dataset, 100, False),
    )


# evaluate a trained model on MNIST data
#
# dataloader    dataloader of examples to evaluate on
# model         trained PyTorch model
# loss_fn       loss function (e.g. torch.nn.CrossEntropyLoss)
#
# returns       tuple of (loss, accuracy), both python floats
@torch.no_grad()
def evaluate_model(dataloader, model, loss_fn):
    loss = 0
    correct = total = 0
    for batch, (features, labels) in enumerate(dataloader):
        probabilities = model(features)
        y_hat = torch.argmax(probabilities, 1)
        loss += loss_fn(probabilities, labels)
        correct += torch.sum(y_hat == labels)
        total += labels.size()[0]
    return (loss.item() / (batch + 1), correct.item() / total)


# build a fully connected two-hidden-layer neural network for MNIST data
#
# returns   a new model of type torch.nn.Sequential
def make_fully_connected_model(input_dim,hidden_dim1,hidden_dim2,output_dim):

    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(input_dim, hidden_dim1),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim1, hidden_dim2),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim2, output_dim),
    )


# build a fully connected two-hidden-layer neural network with Batch Norm
#
# returns   a new model of type torch.nn.Sequential
def make_fully_connected_model_batchnorm(input_dim,hidden_dim1,hidden_dim2,output_dim):

    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(input_dim,hidden_dim1),
        torch.nn.BatchNorm1d(hidden_dim1),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim1,hidden_dim2),
        torch.nn.BatchNorm1d(hidden_dim2),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim2, output_dim),
        torch.nn.BatchNorm1d(output_dim),
    )


# build a convolutional neural network
#
# returns   a new model of type torch.nn.Sequential
def make_cnn_model():

    # 28 x 28 x 1
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, 3, 1),  # 26 x 26 x 16
        torch.nn.ReLU(),  # 26 x 26 x 16
        torch.nn.BatchNorm2d(16),  # 26 x 26 x 16
        torch.nn.ReLU(),  # 26 x 26 x 16
        torch.nn.Conv2d(16, 16, 3, 1),  # 24 x 24 x 16
        torch.nn.ReLU(),  # 24 x 24 x 256
        torch.nn.BatchNorm2d(16),  # 24 x 24 x 16
        torch.nn.ReLU(),  # 24 x 24 x 16
        torch.nn.MaxPool2d(2),  # 12 x 12 x 16
        torch.nn.Conv2d(16, 32, 3, 1),  # 10 x 10 x 32
        torch.nn.ReLU(),  # 10 x 10 x 32
        torch.nn.BatchNorm2d(32),  # 10 x 10 x 32
        torch.nn.ReLU(),  # 10 x 10 x 32
        torch.nn.Conv2d(32, 32, 3, 1),  # 8 x 8 x 32
        torch.nn.ReLU(),  # 8 x 8 x 32
        torch.nn.BatchNorm2d(32),  # 8 x 8 x 32
        torch.nn.ReLU(),  # 8 x 8 x 32
        torch.nn.MaxPool2d(2),  # 4 x 4 x 32
        torch.nn.Flatten(),  # 4 * 4 * 32
        torch.nn.Linear(16 * 32, 128),  # 128
        torch.nn.ReLU(),  # 128
        torch.nn.Linear(128, 10),  # 10
    )


# train a neural network on MNIST data
#
# train_dataloader   training dataloader
# test_dataloader    test dataloader
# model              dnn model to be trained (training should mutate this)
# loss_fn            loss function
# optimizer          an optimizer that inherits from torch.optim.Optimizer
# epochs             number of epochs to run
# eval_train_stats   boolean; whether to evaluate statistics on training set each epoch
# eval_test_stats    boolean; whether to evaluate statistics on test set each epoch
#
# returns   a tuple of
#   train_loss       an array of length `epochs` containing the training loss after each epoch, or [] if eval_train_stats == False
#   train_acc        an array of length `epochs` containing the training accuracy after each epoch, or [] if eval_train_stats == False
#   test_loss        an array of length `epochs` containing the test loss after each epoch, or [] if eval_test_stats == False
#   test_acc         an array of length `epochs` containing the test accuracy after each epoch, or [] if eval_test_stats == False
#   approx_tr_loss   an array of length `epochs` containing the average training loss of examples processed in this epoch
#   approx_tr_acc    an array of length `epochs` containing the average training accuracy of examples processed in this epoch
def train(
    train_dataloader,
    test_dataloader,
    model,
    loss_fn,
    optimizer,
    epochs,
    eval_train_stats=True,
    eval_test_stats=True,
):
    
    res = ([], [], [], [], [], [])
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for batch, (features, labels) in enumerate(train_dataloader):
            probabilities = model(features)
            y_hat = torch.argmax(probabilities, 1)
            loss = loss_fn(probabilities, labels)
            acc = torch.sum(y_hat == labels) / labels.size()[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += acc.item()

        res[0].append(running_loss / (batch + 1))
        res[1].append(running_acc / (batch + 1))

        model.eval()
        if eval_train_stats:
            (loss, accuracy) = evaluate_model(train_dataloader, model, loss_fn)
            res[2].append(loss)
            res[3].append(accuracy)
        if eval_test_stats:
            (loss, accuracy) = evaluate_model(test_dataloader, model, loss_fn)
            res[4].append(loss)
            res[5].append(accuracy)
            print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    return res


def create_plots(name,results,n_epochs):
    
    fig, ax = plt.subplots()
    epoch_range = np.arange(1, n_epochs + 1)
    ax.plot(epoch_range, results[0], label="Training Loss (Minibatch Average)")
    ax.plot(epoch_range, results[2], label="End-of-Epoch Training Losses")
    ax.plot(epoch_range, results[4], label="Test Loss")

    ax.legend()
    ax.set_title(f"{name} Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(f"fig/{name}_losses.jpg")
    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(epoch_range, results[1], label="Training Accuracy (Minibatch Average)")
    ax.plot(epoch_range, results[3], label="End-of-Epoch Training Accuracy")
    ax.plot(epoch_range, results[5], label="Test Accuracy")

    ax.legend()
    ax.set_title(f"{name} Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    plt.savefig(f"fig/{name}_accuracy.jpg")
    plt.clf()
    
    
if __name__ == "__main__":
    (train_dataset, test_dataset) = load_MNIST_dataset()

    B = 100
    (train_dataloader, test_dataloader) = construct_dataloaders(train_dataset, test_dataset, B)

    
    # Basic Hyperparameters
    n_epochs = 10
    input_dim = 784
    hidden_dim1 = 1024
    hidden_dim2 = 256
    output_dim = 10
    
    print("Running SGD")
    alpha = 0.1
    
    model = make_fully_connected_model(input_dim,hidden_dim1,hidden_dim2,output_dim)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=alpha)
    
    start = time.time()
    res = train(train_dataloader, test_dataloader, model, loss_fn, optim, n_epochs, True, True)
    end = time.time()

    print(f"Total time: {end - start:.2f}s")

    create_plots("SGD",res,n_epochs)

    
    
    print("Running SGD with Momentum")
    alpha = 0.1
    beta = 0.9
    
    model = make_fully_connected_model(input_dim,hidden_dim1,hidden_dim2,output_dim)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=alpha, momentum=beta)
    
    start = time.time()
    res = train(train_dataloader, test_dataloader, model, loss_fn, optim, n_epochs, True, True)
    end = time.time()
    print(f"Total time: {end - start:.2f}s")

    create_plots("SGD_momentum",res,n_epochs)

    
    
    print("Running ADAM")
    alpha = 0.001
    rho1 = 0.99
    rho2 = 0.999
    
    model = make_fully_connected_model(input_dim,hidden_dim1,hidden_dim2,output_dim)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=alpha, betas=(rho1, rho2))
    
    start = time.time()
    res = train(train_dataloader, test_dataloader, model, loss_fn, optim, n_epochs, True, True)
    end = time.time()
    print(f"Total time: {end - start:.2f}s")

    create_plots("ADAM",res,n_epochs)


    
    print("Running SGD with Batch Normalization")
    alpha = 0.001
    beta = 0.9
    
    model = make_fully_connected_model_batchnorm(input_dim,hidden_dim1,hidden_dim2,output_dim)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=alpha, momentum=beta)
    
    start = time.time()
    res = train(train_dataloader, test_dataloader, model, loss_fn, optim, n_epochs, True, True)
    end = time.time()
    print(f"Total time: {end - start:.2f}s")

    create_plots("SGD_batchnorm",res,n_epochs)


    
    print("Running CNN")
    alpha = 0.001
    rho1 = 0.99
    rho2 = 0.999
    
    model = make_cnn_model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=alpha, betas=(rho1, rho2))
    
    start = time.time()
    res = train(train_dataloader, test_dataloader, model, loss_fn, optim, n_epochs, True, True)
    end = time.time()
    print(f"Total time: {end - start:.2f}s")

    create_plots("CNN",res,n_epochs)
    
    
    
    
    # Search over learning rate
    alphas = [1.0,0.3,0.1,0.03,0.01,0.003,0.001]
    loss_fn = torch.nn.CrossEntropyLoss()
    beta = 0.9
    
    for alpha in alphas:
        
        print(f"SGD with alpha = {alpha}")
        model = make_fully_connected_model(input_dim,hidden_dim1,hidden_dim2,output_dim)

        
        optim = torch.optim.SGD(model.parameters(), lr=alpha, momentum=beta)
        
        start = time.time()
        res = train(train_dataloader, test_dataloader, model, loss_fn, optim, n_epochs, False, False)
        end = time.time()
    
        print(f"Total time: {end - start:.2f}s")
        
        model.eval()
        (test_loss,test_accuracy) = evaluate_model(test_dataloader, model, loss_fn)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")
      
    
       
    
    # Grid Search Hyperparameter Optimization
    alphas = [0.1,0.01]
    betas = [0.9,0.99]
    layer_widths = [512,1024,2048]
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for alpha in alphas:
        for beta in betas:
            for width in layer_widths:
                
                print(f"SGD with alpha = {alpha}, beta = {beta}, and width = {width}")

                model = make_fully_connected_model(input_dim,width,hidden_dim2,output_dim)
                optim = torch.optim.SGD(model.parameters(), lr=alpha, momentum=beta)

                start = time.time()
                res = train(train_dataloader, test_dataloader, model, loss_fn, optim, n_epochs, False, False)
                end = time.time()

                print(f"Total time: {end - start:.2f}s")

                model.eval()
                (train_loss,train_accuracy) = evaluate_model(train_dataloader, model, loss_fn)
                (test_loss,test_accuracy) = evaluate_model(test_dataloader, model, loss_fn)
                print(f"Train Loss: {train_loss}")
                print(f"Train Accuracy: {train_accuracy}")
                print(f"Test Loss: {test_loss}")
                print(f"Test Accuracy: {test_accuracy}")
    

    # Random Search Hyperparameter Optimization
    n_samples = 12
    loss_fn = torch.nn.CrossEntropyLoss()
    
    alphas = np.random.uniform(0.01,0.1,size=(n_samples,))
    betas = np.random.uniform(0.9,0.99,size=(n_samples,))
    layer_widths = np.random.randint(512,2048,size=(n_samples,))
    

    

    for alpha,beta,width in zip(alphas,betas,layer_widths):
        
        print(f"SGD with alpha = {alpha}, beta = {beta}, and width = {width}")

        model = make_fully_connected_model(input_dim,width,hidden_dim2,output_dim)
        optim = torch.optim.SGD(model.parameters(), lr=alpha, momentum=beta)

        start = time.time()
        res = train(train_dataloader, test_dataloader, model, loss_fn, optim, n_epochs, False, False)
        end = time.time()

        print(f"Total time: {end - start:.2f}s")

        model.eval()
        (train_loss,train_accuracy) = evaluate_model(train_dataloader, model, loss_fn)
        (test_loss,test_accuracy) = evaluate_model(test_dataloader, model, loss_fn)
        print(f"Train Loss: {train_loss}")
        print(f"Train Accuracy: {train_accuracy}")
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")
    