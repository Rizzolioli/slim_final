import numpy as np
import torch


def linear_increase(data_size, n_gens, X_train, y_train):

    # starting at 10% of the data
    start = np.ceil(data_size * 0.1)

    # ending at the full dataset
    end = data_size

    # getting a list with the batch size to be used at each generation
    batch_sizes = [int(bs) for bs in np.linspace(start, end, n_gens + 1)]

    def li(generation):
        # getting the batch size for the corresponding generation
        bs = batch_sizes[generation]
        # getting the indices of the training instances that are to be used
        indices = torch.randperm(X_train.size(0))[:bs]

        return X_train[indices], y_train[indices]

    return li


def linear_decrease(data_size, n_gens, X_train, y_train):
    # starting at 100% of the data
    start = data_size

    # ending at 1 element
    end = 1

    # getting a list with the batch size to be used at each generation
    batch_sizes = [int(bs) for bs in np.linspace(start, end, n_gens + 1)]

    def ld(generation):

        # getting the batch size for the corresponding generation
        bs = batch_sizes[generation]

        # getting the indices of the training instances that are to be used
        indices = torch.randperm(X_train.size(0))[:bs]

        return X_train[indices], y_train[indices]

    return ld