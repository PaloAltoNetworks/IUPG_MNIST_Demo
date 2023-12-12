"""
IUPG_Input.py

Contains a class that organizes input data and facilitates ease of use
with training and validation an IUPG model.

@author: Brody Kutt (bkutt@paloaltonetworks.com)
Copyright (c) 2020 Palo Alto Networks
"""

import random
import numpy as np


class IUPG_Input:
    def __init__(self,
                 X,
                 y=None,
                 X_val=[],
                 y_val=[],
                 n_classes=11,
                 random_seed=1994):

        self.using_val = len(X_val) > 0

        self.X = X  # All the training images
        self.y = y  # All the class indices
        self.n_samples = X.shape[0]

        self.i = 0  # Index for batch sampling
        self.num_epochs = 0  # How many epochs processed so far
        self.n_classes = n_classes
        self.random_seed = random_seed  # Random seed for repeatability
        random.seed(random_seed)
        np.random.seed(random_seed)

        # These variables are updated once .split() is called...
        self.X_train = None
        self.y_train = None  # Training true classes; one hot encoded
        self.X_val = X_val
        self.y_val = y_val  # Testing true classes; one hot encoded
        self.n_train = None  # Calculate this once
        self.n_val = None  # Calculate this once
        if self.using_val:
            self.split()

    def scramble(self):
        """
        Randomizes the training data row orders.
        """
        # Randomize training data
        new_order = random.sample(list(range(self.n_train)), self.n_train)
        self.X_train = self.X_train[new_order]
        self.y_train = self.y_train[new_order, :]

    def one_hot(self, y):
        """
        For internal use only. Convert an array of classes into a one-hot
        encoded version.
        """
        y_one_hot = np.zeros((len(y), self.n_classes))
        y_one_hot[np.arange(len(y)), y] = 1
        return y_one_hot

    def split(self):
        """
        Split X and y into stratified training and validation sets with the
        given percentage of data going into the validation set.
        """
        assert self.using_val
        self.X_train = self.X
        # Turn y matrices into one hot encoded
        self.y_train = self.one_hot(self.y)
        self.y_val = self.one_hot(self.y_val)
        # Update lengths
        self.n_train, self.n_val = len(self.y_train), len(self.y_val)

    def get_val_batch_iter(self, batch_size):
        """
        Generates a batch iterator for the validation dataset. The iterator
        stops once all data has been seen once.
        """
        assert self.using_val
        k = 0  # Keep track of where we are in the loop
        while k < self.n_val:
            batch = {}
            if k + batch_size >= self.n_val:
                batch["X"] = self.X_val[k:self.n_val, :, :]
                batch["y"] = self.y_val[k:self.n_val, :]
                k += batch_size
                yield batch
            else:
                batch["X"] = self.X_val[k:k + batch_size, :, :]
                batch["y"] = self.y_val[k:k + batch_size, :]
                k += batch_size
                yield batch

    def get_train_batch_iter(self, batch_size):
        """
        Generates a batch iterator for the training dataset. The iterator
        stops once all data has been seen once.
        """
        assert self.using_val
        k = 0  # Keep track of where we are in the loop
        while k < self.n_train:
            batch = {}
            if k + batch_size >= self.n_train:
                batch["X"] = self.X_train[k:self.n_train, :, :]
                batch["y"] = self.y_train[k:self.n_train, :]
                k += batch_size
                yield batch
            else:
                batch["X"] = self.X_train[k:k + batch_size, :, :]
                batch["y"] = self.y_train[k:k + batch_size, :]
                k += batch_size
                yield batch

    def get_X_batch_iter(self, batch_size):
        """
        Generates a batch iterator for X. Use this when you are working
        with unlabeled data.
        """
        k = 0  # Keep track of where we are in the loop
        while k < self.n_samples:
            batch = {}
            if k + batch_size >= self.n_samples:
                batch["X"] = self.X[k:self.n_samples, :, :]
                k += batch_size
                yield batch
            else:
                batch["X"] = self.X[k:k + batch_size, :, :]
                k += batch_size
                yield batch

    def get_Xy_batch_iter(self, batch_size):
        """
        Generates a batch iterator for X and y. Use this when you are working
        with no train/val splits.
        """
        k = 0  # Keep track of where we are in the loop
        y_one_hot = self.one_hot(self.y)
        while k < self.n_samples:
            batch = {}
            if k + batch_size >= self.n_samples:
                batch["X"] = self.X[k:self.n_samples, :, :]
                batch["y"] = y_one_hot[k:self.n_samples, :]
                k += batch_size
                yield batch
            else:
                batch["X"] = self.X[k:k + batch_size, :, :]
                batch["y"] = y_one_hot[k:k + batch_size, :]
                k += batch_size
                yield batch

    def next_batch(self, batch_size):
        """
        Returns a batch of training data of the given size and updates index.
        """
        assert self.using_val
        batch = {}
        if self.i + batch_size >= self.n_train:
            # Grab the rest of the data that hasn't been seen yet
            X = self.X_train[self.i:self.n_train, :, :]
            y = self.y_train[self.i:self.n_train, :]
            # Increment number of epochs
            self.num_epochs += 1
            # Randomize rows after each epoch
            self.scramble()
            # Grab enough data from the new epoch for appropriate batch size
            diff = (self.i + batch_size) - self.n_train
            batch["X"] = np.concatenate((X, self.X_train[0:diff, :, :]))
            batch["y"] = np.vstack((y, self.y_train[0:diff, :]))
            self.i = diff
        else:
            batch["X"] = self.X_train[self.i:self.i + batch_size, :, :]
            batch["y"] = self.y_train[self.i:self.i + batch_size, :]
            self.i += batch_size
        return batch

    def get_rnd_train_sample(self, size):
        """
        Randomly sample the training set.
        """
        assert self.using_val
        rnd_sample = {}
        new_order = random.sample(list(range(self.n_train)), self.n_train)
        new_order = new_order[:size]
        rnd_sample["X"] = self.X_train[new_order, :, :]
        rnd_sample["y"] = self.y_train[new_order, :]
        return rnd_sample

    def get_balanced_rnd_train_sample(self, size):
        """
        Randomly sample the training set that is balanced w.r.t. class
        frequencies.
        """
        assert self.using_val
        rnd_sample = self.get_rnd_train_sample(self.n_train)
        bal_order = []
        n_noise = 0
        n_digit = [0 for i in range(self.n_classes - 1)]
        noise_limit = int(size / 2.0)
        digit_limit = int(noise_limit / float(self.n_classes - 1))
        for i, y in enumerate(rnd_sample["y"]):
            y_1d = np.argmax(y)
            if y_1d == 0 and n_noise < noise_limit:
                n_noise += 1
                bal_order.append(i)
            elif y_1d > 0 and n_digit[y_1d - 1] < digit_limit:
                n_digit[y_1d - 1] += 1
                bal_order.append(i)

        bal_rnd_sample = {}
        bal_rnd_sample["X"] = rnd_sample["X"][bal_order]
        bal_rnd_sample["y"] = rnd_sample["y"][bal_order, :]
        return bal_rnd_sample

    def get_batch_iter(self, sample, batch_size):
        """
        Generate an batch iterator for a sample.
        """
        sample_size = len(sample["y"])
        k = 0  # Keep track of where we are in the loop
        while k < sample_size:
            batch = {}
            if k + batch_size >= sample_size:
                batch["X"] = sample["X"][k:sample_size, :, :]
                batch["y"] = sample["y"][k:sample_size, :]
                k += batch_size
                yield batch
            else:
                batch["X"] = sample["X"][k:k + batch_size, :, :]
                batch["y"] = sample["y"][k:k + batch_size, :]
                k += batch_size
                yield batch
