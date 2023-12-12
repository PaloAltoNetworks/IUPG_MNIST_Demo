"""
IUPG_Builder.py

MNIST classifier network builder script which uses the
"innocent until proven guilty" learning scheme.

The main public functions for instances to call is train() and, optionally,
graph_summaries()

@author: Brody Kutt (bkutt@paloaltonetworks.com)
Copyright (c) 2020 Palo Alto Networks
"""

import os
import csv
import time
import utils
import matplotlib
matplotlib.use("Agg")  # Force no use of Xwindows backend
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from progress.bar import Bar
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt
from IUPG_Input import IUPG_Input
from sklearn.manifold import TSNE
import matplotlib.font_manager as fm
from timeit import default_timer as timer
from matplotlib.collections import QuadMesh
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.python.client import device_lib

# Set this to true to debug GPU usage
LOG_DEVICE_PLACEMENT = False

# The format of the CSV summary file to be written during batch training
BATCH_SUMMARY_FORMAT = ["TIME", "STEP", "BATCH_LOSS"]
# The format of the CSV summary file to be written during batch training
TRAIN_SUMMARY_FORMAT = [
    "TIME",
    "STEP",
    "TRAIN_LOSS",
    "TRAIN_ACC",
    "PROTOTYPE_COUNTS",
]
# The format of the CSV summary file to be written during validation
VAL_SUMMARY_FORMAT = [
    "TIME", "STEP", "VAL_LOSS", "VAL_ACC", "PROTOTYPE_COUNTS"
]

CNN_PARAMS = [
    "C_FILT_SIZES",
    "C_NUM_FILTS",
    "DEEP_FILT_SIZES",
    "DEEP_NUM_FILTS",
    "C_DEEP_POOL",
]
FC_PARAMS = ["FC_NUM_FILTS"]

# The format of the CSV summary file to be written at the end of training
FINAL_SUMMARY_FORMAT = (
    [
        "NAME",
        "TRAIN_BATCH_SIZE",
        "DROPOUT_KEEP_PROB",
        "STEPS_PER_VAL_EVAL",
        "MAX_EVALS_SINCE_OVERWRITE",
        "OPTIMIZER",
        "L2_LAMBDA",
        "LEARNING_RATE",
        "GAMMA",
        "N_PROTOS",
        "OUT_DIM",
        "DIST_METRIC",
    ] + CNN_PARAMS + FC_PARAMS +
    ["STEPS_UNTIL_BEST", "BEST_VAL_LOSS", "BEST_VAL_ACC", "SEC_UNTIL_10000"])

# How dates and times appear
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# The batch size when doing evaluation.
EVAL_BATCH_SIZE = 32


def print_cm(cm, labels, normalize=True):
    """
    Print a confusion matrix.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    columnwidth = max([len(x) for x in labels] + [8])  # 8 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label, end=" ")
        for j in range(len(labels)):
            if normalize:
                if not np.isnan(cm[i, j]):
                    cell = "%{0}.3f".format(columnwidth) % cm[i, j]
                else:
                    cell = (" " * (columnwidth - 1)) + "-"
            else:
                cell = "%{0}d".format(columnwidth) % cm[i, j]
            print(cell, end=" ")
        print()


def print_pps(pps, stat_name, labels):
    """
    Print the "per proto stats." For each prototype, this allows us to peek
    at the min/max/avg/stdev of distances to data of each class.
    """
    columnwidth = max([len(x) for x in labels] + [8])  # 8 is value length
    empty_cell = " " * columnwidth
    # print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # print rows
    for i, label in enumerate(labels):
        if i == 0:
            continue
        else:
            print("    %{0}s".format(columnwidth) % label, end=" ")
            for j in range(len(labels)):
                if "%s_%d_dist" % (stat_name, j) in pps[i]:
                    cell = "%{0}.3f".format(columnwidth) % pps[i][
                        "%s_%d_dist" % (stat_name, j)]
                else:
                    cell = (" " * (columnwidth - 1)) + "-"
                print(cell, end=" ")
            print()


class IUPG_Builder(object):
    def __init__(
        self,
        # The identifiable name of the model that will be saved once this
        # instance has undergone a training process. The "winner" model
        # with the lowest loss will be the one who is saved with this name.
        name,
        # Base path to save all models and training results.
        base_save_path,
        # The params which define the convolutional layers.
        conv_params,
        # The params which define the fully connected layers.
        fc_params,
        # Size of the output layer. This defines the dimensionalty of the
        # output vector space from which distances will be calculated.
        out_dim,
        # The type of distance function to use (sq_err, abs_err, or
        # siamese).
        dist_metric="siamese",
        # An optional way to specify initializations of the prototypes
        # as opposed to random initializations.
        proto_inits=None,
        # If prototype initializations is set to random, this is the
        # maximum magnitude of the random values.
        rnd_proto_init_mag=1.0,
        # The coefficient for the L2 loss. Use l2_lambda=0.0 for no
        # regularization.
        l2_lambda=0.0,
        # The gamma parameter which controls the weight of the targeted
        # class in the IUPG loss function.
        gamma=1.0,
        # Seed all random generation for repeatability.
        random_seed=1994,
        # A dictionary to map prototype numbers to interpretable string
        # labels.
        int2label=None,
        # Whether to use global max pooling after final convolutional
        # layers
        global_max_pool=True,
    ):

        self.name = name
        self.check_conv_params(conv_params)
        self.conv_p = conv_params
        self.fc_p = fc_params

        self.base_save_path = base_save_path
        self.out_dim = out_dim
        self.dist_metric = dist_metric
        if dist_metric == "sq_err":
            self.dist_func = self.pairwise_sq_err
        elif dist_metric == "abs_err":
            self.dist_func = self.pairwise_abs_err
        elif dist_metric == "siamese":
            self.dist_func = self.pairwise_siamese_dist
        else:
            assert dist_metric in ["sq_err", "abs_err", "siamese"]
        self.l2_lambda = l2_lambda
        self.gamma = gamma
        self.int2label = int2label
        self.random_seed = random_seed
        self.sec_until_10000 = -1
        self.cur_device_idx = 0
        self.device_type, self.device_ids = self.get_avail_devices()
        self.n_devices = len(self.device_ids)
        self.epsilon = 1e-8
        self.proto_inits = proto_inits
        self.global_max_pool = global_max_pool
        self.rnd_proto_init_mag = rnd_proto_init_mag
        self.use_rnd_proto_init = True
        if self.proto_inits is not None:
            assert self.proto_inits.shape[0] == utils.n_protos()
            assert self.proto_inits.shape[1] == utils.im_height()
            assert self.proto_inits.shape[2] == utils.im_width()
            self.use_rnd_proto_init = False

    def check_conv_params(self, p):
        """
        Do some error checking on a dict of convolutional params, p.
        """
        assert len(p["filt_sizes"]) == len(p["num_filts"])
        for size in p["filt_sizes"]:
            assert size % 2 != 0  # Ensure its an odd number
        assert len(p["deep_filt_sizes"]) == len(p["filt_sizes"])
        assert len(p["deep_num_filts"]) == len(p["filt_sizes"])
        assert len(p["deep_pool"]) == len(p["filt_sizes"])
        zero_sizes = np.where(p["deep_filt_sizes"] == 0)[0]
        for i in zero_sizes:
            assert p["deep_num_filts"][i] == 0
            assert not p["deep_pool"][i]
        zero_nums = np.where(p["deep_num_filts"] == 0)[0]
        for i in zero_nums:
            assert p["deep_filt_sizes"][i] == 0
            assert not p["deep_pool"][i]
        non_zero_sizes = np.array(p["deep_filt_sizes"])[np.nonzero(
            p["deep_filt_sizes"])]
        for size in non_zero_sizes:
            assert size % 2 != 0  # Ensure its an odd number

    def get_avail_devices(self):
        """
        Return a list of all available devices, preferring GPUs if available.
        Note that this is referring to number of CPUs not number of possible
        cores/threads.
        """
        local_device_protos = device_lib.list_local_devices()
        gpus = [x.name for x in local_device_protos if x.device_type == "GPU"]
        cpus = [x.name for x in local_device_protos if x.device_type == "CPU"]
        device_ids = []
        if len(gpus) > 0:
            for device_str in gpus:
                device_ids.append(device_str.split(":")[-1])
            device_type = "gpu"
        else:
            for device_str in cpus:
                device_ids.append(device_str.split(":")[-1])
            device_type = "cpu"
        return device_type, device_ids

    def next_device(self):
        """
        Get the name of the next device to assign an operation to.
        """
        device_name = "/%s:%d" % (self.device_type, self.cur_device_idx)
        self.cur_device_idx = (self.cur_device_idx + 1) % max(
            self.n_devices, 1)
        return device_name

    def pairwise_abs_err(self, A, B):
        """
        Computes pairwise L1 distances between each element of A and each
        element of B.

        Args:
            A,    [m, d] matrix
            B,    [n, d] matrix
        Returns:
            D,    [m, n] matrix of pairwise absolute errors
        """
        with tf.compat.v1.variable_scope("pair_abs_err"), tf.device(
                self.next_device()):
            xx = tf.expand_dims(A, -1)
            xx = tf.tile(xx, tf.stack([1, 1, tf.shape(B)[0]]))

            yy = tf.expand_dims(B, -1)
            yy = tf.tile(yy, tf.stack([1, 1, tf.shape(A)[0]]))
            yy = tf.transpose(yy, perm=[2, 1, 0])
            return tf.reduce_sum(tf.abs(xx - yy), 1)

    def pairwise_sq_err(self, A, B):
        """
        Computes pairwise squared error between each element of A and each
        element of B.

        Args:
            A,    [m, d] matrix
            B,    [n, d] matrix
        Returns:
            D,    [m, n] matrix of pairwise squared errors
        """
        with tf.compat.v1.variable_scope("pair_sq_err"), tf.device(
                self.next_device()):
            # squared norms of each row in A and B
            na = tf.reduce_sum(tf.square(A), 1)
            nb = tf.reduce_sum(tf.square(B), 1)

            # na as a row and nb as a column vectors
            na = tf.reshape(na, [-1, 1])
            nb = tf.reshape(nb, [1, -1])

            # return pairwise squared euclidead difference matrix
            return tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0)

    def pairwise_siamese_dist(self, A, B):
        """
        Computes pairwise "siamese" L1 distance with learnable alpha vector.

        Args:
            A,    [m, d] matrix
            B,    [n, d] matrix
        Returns:
            D,    [m, n] matrix of pairwise squared errors
        """
        with tf.compat.v1.variable_scope("pair_siam_dist"), tf.device(
                self.next_device()):
            xx = tf.expand_dims(A, -1)
            xx = tf.tile(xx, tf.stack([1, 1, tf.shape(B)[0]]))

            yy = tf.expand_dims(B, -1)
            yy = tf.tile(yy, tf.stack([1, 1, tf.shape(A)[0]]))
            yy = tf.transpose(yy, perm=[2, 1, 0])
            zz = tf.reduce_sum(tf.abs(xx - yy) * tf.exp(self.alpha), 1)
            return zz

    def adj_sigmoid(self, X):
        """
        Maps values [0, inf) -> [0, 1)
        """
        return (2.0 / (1.0 + tf.exp(-2 * X))) - 1.0

    def iupg_loss(self, D, y_shaved, omega_vecs, gamma=1.0):
        """
        The generalized IUPG loss function.

        Args:
            D,           [batch_size, n_protos] ndarray, of distances.
            y_shaved,    [batch_size, n_target_classes] ndarray, of labels. Assumes the first
                         column (the noise class) has been shaved off of this
                         one-hot encoded matrix.
            omega_vecs,  [n_target_classes, n_protos] ndarray, of omega vectors.
            gamma,       float, specifies the weight of the true class.
        Returns:
            losses,      [batch_size] ndarray, of losses for each sample.
        """
        # Stack distance and omega vectors so they can be multiplied
        stacked_omega_vecs = tf.reshape(
            tf.tile(omega_vecs, [self.batch_size_ref, 1]),
            [self.batch_size_ref,
             utils.n_target_classes(),
             utils.n_protos()],
        )
        stacked_D = tf.reshape(
            tf.tile(D, [1, utils.n_target_classes()]),
            [self.batch_size_ref,
             utils.n_target_classes(),
             utils.n_protos()],
        )
        # Linearly shift all values in D according to omega vectors
        trans_D = (stacked_D + self.epsilon) * (1.0 / stacked_omega_vecs)
        # Reduce minimum to bring to [batch_size, n_target_classes]. For each
        # sample, this fetches the minimum distance among prototypes
        # designated to each target class.
        min_trans_D = tf.reduce_min(trans_D, axis=2)
        # Clip values to specified ranges
        clipped_D = tf.clip_by_value(min_trans_D, self.epsilon, 1.0)
        clipped_D_inv = tf.clip_by_value((1 - min_trans_D), self.epsilon, 1.0)
        # Compute all values of sum over classes for each sample
        inner_sum = y_shaved * -tf.math.log(clipped_D_inv) + (1 - y_shaved) * (
            1.0 / gamma) * -tf.math.log(clipped_D)
        return tf.reduce_sum(inner_sum, axis=1)

    def _build_IUPG(self):
        """
        Build the IUPG model as a TF graph.
        """
        # Set the random seed in TF for repeatability
        tf.compat.v1.random.set_random_seed(self.random_seed)

        # Placeholder for batch of images
        self.X_batch = tf.compat.v1.placeholder(
            tf.float32,
            # [Batch size, Image Height, Image Width]
            [None, utils.im_height(), utils.im_width()],
            name="X_batch",
        )
        self.batch_size_ref = tf.shape(self.X_batch)[0]

        # Placeholder for true labels of the batch
        self.y_batch = tf.compat.v1.placeholder(
            tf.float32, [None, utils.n_classes_total()], name="y_batch")
        self.y_batch_1d = tf.cast(tf.argmax(self.y_batch, 1),
                                  dtype=tf.float32,
                                  name="y_batch_1d")
        self.y_shaved = self.y_batch[:, 1:]  # Shave off noise class

        # Vectors that define prototype to class designations. For this demo, each
        # target class simply has 1 prototype assigned to it.
        self.omega_vecs = (tf.eye(
            num_rows=utils.n_target_classes(),
            num_columns=utils.n_protos(),
            dtype=tf.float32,
        ) + self.epsilon)

        # Placeholder for probability of keeping a neuron in the dropout layer
        self.drop_keep_prob = tf.compat.v1.placeholder(tf.float32,
                                                       name="drop_keep_prob")
        # Flag to signal sending the sample input through the network
        self.run_samps = tf.compat.v1.placeholder_with_default(
            True, shape=[], name="run_samps")
        # Flag to signal sending the prototypes through the network
        self.run_protos = tf.compat.v1.placeholder_with_default(
            True, shape=[], name="run_protos")
        # Value of provided prototypes in U space when in testing mode,
        # otherwise, we compute them
        self.given_U_protos = tf.compat.v1.placeholder_with_default(
            tf.zeros([utils.n_protos(), self.out_dim], dtype=tf.float32),
            shape=[utils.n_protos(), self.out_dim],
            name="given_U_protos"
        )
        # Learnable vector of weights for distance metric
        self.alpha = tf.Variable(tf.zeros([1, self.out_dim, 1]), name="alpha")
        # Stores weights that are subjected to L2 norm
        self.W_for_l2 = []

        with tf.name_scope("proto_init") as scope:
            # Define initial values of prototypes
            if self.use_rnd_proto_init:
                protos = self.rnd_proto_init()
            else:
                protos = self.proto_init()
            self.protos = tf.identity(protos, name="protos")

        # Add trivial dimension to bring tensor into 4D for conv layers
        self.X_batch_ex = tf.expand_dims(self.X_batch, -1, name="X_batch_ex")
        self.protos_ex = tf.expand_dims(self.protos, -1, name="protos_ex")

        with tf.compat.v1.variable_scope("network") as scope:
            # Send input through the network, if needed
            samp_U_vec = tf.cond(
                self.run_samps,
                lambda: self.network(self.X_batch_ex),
                lambda: tf.zeros(shape=[self.batch_size_ref, self.out_dim],
                                 dtype=tf.float32),
            )  # Dummy value
            self.samp_U_vec = tf.identity(samp_U_vec, name="samp_U_vec")
            # Ensure that the model variables are reused
            scope.reuse_variables()
            # Send prototypes through the network, if needed
            proto_U_vec = tf.cond(
                self.run_protos,
                lambda: self.network(self.protos_ex),
                lambda: self.given_U_protos,
            )
            self.proto_U_vec = tf.identity(proto_U_vec, name="proto_U_vec")

        with tf.name_scope("output"):
            # Calculate distances to prototypes
            D = self.dist_func(self.samp_U_vec, self.proto_U_vec)
            self.D = tf.identity(D, name="D")
            # Send distances through adjusted sigmoid function
            sig_D = self.adj_sigmoid(self.D)
            self.sig_D = tf.identity(sig_D, name="sig_D")
            # Calculate distance to best fit prototype vec
            self.min_D = tf.math.reduce_min(self.sig_D, axis=1, name="min_D")
            # Calculate the winning prototypes
            self.win_protos = tf.math.argmin(
                self.sig_D, axis=1, name="win_protos") + 1

        # Calculate the loss
        with tf.name_scope("avg_loss"):
            losses = self.iupg_loss(self.sig_D,
                                    self.y_shaved,
                                    self.omega_vecs,
                                    gamma=self.gamma)
            self.losses = tf.identity(losses, name="losses")
            l2_loss = np.sum([tf.nn.l2_loss(W) for W in self.W_for_l2])
            self.avg_loss = tf.reduce_mean(self.losses, name="avg_loss")
            self.avg_loss_l2 = self.avg_loss + (self.l2_lambda * l2_loss)

    def network(self, inp):
        """
        Defines the main network encoder.
        """
        # Send through convolutional layers
        Z_flat = self.conv_layers(inp)

        # Perform dropout on normalized activations
        Z_flat_dropped = tf.nn.dropout(Z_flat,
                                       rate=1 - self.drop_keep_prob,
                                       name="Z_flat_dropped")

        # Send through fully connected layers
        last_inp, last_dim = self.fc_layers(Z_flat_dropped)

        # Send through final projection to U space
        return self.U_project(last_inp, last_dim)

    def conv_layers(self, inp):
        """
        Create parallel convolutional layers.
        """
        all_pooled_outputs = []
        # For each parallel conv layer...
        conv_layer_id = 1
        for fs1, fn1, fs2, fn2, p2 in zip(
                self.conv_p["filt_sizes"],
                self.conv_p["num_filts"],
                self.conv_p["deep_filt_sizes"],
                self.conv_p["deep_num_filts"],
                self.conv_p["deep_pool"],
        ):
            unique_layer_id = "%d.%s" % (conv_layer_id, fs1)
            scope = "conv-layer-%s" % unique_layer_id
            with tf.device(self.next_device()), tf.name_scope(scope):
                # Define the convolution weights
                filter_shape = [fs1, fs1, 1, fn1]
                W_c1 = tf.compat.v1.get_variable(
                    "W_c1-%s" % unique_layer_id,
                    initializer=lambda: tf.random.truncated_normal(
                        filter_shape, stddev=1e-2),
                    trainable=True,
                )
                b_c1 = tf.compat.v1.get_variable(
                    "b_c1-%s" % unique_layer_id,
                    initializer=lambda: tf.random.truncated_normal(
                        [fn1], mean=0.5, stddev=1e-2),
                    trainable=True,
                )

                # Do the first convolution
                conv1 = tf.nn.conv2d(
                    inp,
                    W_c1,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv1-%s" % unique_layer_id,
                )

                # Add bias to activations
                conv_add_b1 = tf.nn.bias_add(conv1,
                                             b_c1,
                                             name="conv_add_b1-%s" %
                                             unique_layer_id)

                if fs2 == 0 or fn2 == 0:
                    # Apply activation function and be done now
                    Z = tf.nn.relu(conv_add_b1, name="Z-%s" % unique_layer_id)
                else:
                    # Perform one more convolution on Z_pre
                    Z_pre = tf.nn.relu(conv_add_b1,
                                       name="Z_pre-%s" % unique_layer_id)
                    # Do 2x2 max pooling, if required
                    if p2:
                        Z_pre = tf.nn.max_pool2d(
                            input=Z_pre,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding="SAME",
                            name="Z_pre_pooled-%s" % unique_layer_id,
                        )
                    # Define weights and biases for second convolution
                    unique_layer_id = "%d.%s" % (conv_layer_id, fs2)
                    filter_shape = [fs2, fs2, fn1, fn2]
                    W_c2 = tf.compat.v1.get_variable(
                        "W_c2-%s" % unique_layer_id,
                        initializer=lambda: tf.random.truncated_normal(
                            filter_shape, stddev=1e-2),
                        trainable=True,
                    )
                    b_c2 = tf.compat.v1.get_variable(
                        "b_c2-%s" % unique_layer_id,
                        initializer=lambda: tf.random.truncated_normal(
                            [fn2], mean=0.5, stddev=1e-2),
                        trainable=True,
                    )

                    # Do second convolution
                    conv2 = tf.nn.conv2d(
                        Z_pre,
                        W_c2,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                        name="conv2-%s" % unique_layer_id,
                    )
                    # Add in second biases
                    conv_add_b2 = tf.nn.bias_add(conv2,
                                                 b_c2,
                                                 name="conv_add_b2-%s" %
                                                 unique_layer_id)
                    # Apply activation function on deep input
                    Z = tf.nn.relu(conv_add_b2,
                                   name="deep_Z-%s" % unique_layer_id)

                Z_shape = Z.get_shape().as_list()
                if self.global_max_pool:
                    # Global max pooling over the activation maps
                    Z_final = tf.math.reduce_max(
                        Z,
                        axis=(1, 2),
                        keepdims=True,
                        name="Z_global_max-%s" % unique_layer_id,
                    )
                else:
                    unique_layer_id = "%d.%s" % (conv_layer_id, "FCC")
                    # Implement the final fully connected convolutional layer
                    final_filter_shape = [
                        Z_shape[1],
                        Z_shape[2],
                        Z_shape[3],
                        Z_shape[3],
                    ]
                    W_c3 = tf.compat.v1.get_variable(
                        "W_c3-%s" % unique_layer_id,
                        initializer=lambda: tf.random.truncated_normal(
                            final_filter_shape, stddev=1e-2),
                        trainable=True,
                    )
                    b_c3 = tf.compat.v1.get_variable(
                        "b_c3-%s" % unique_layer_id,
                        initializer=lambda: tf.random.truncated_normal(
                            [Z_shape[3]], mean=0.5, stddev=1e-2),
                        trainable=True,
                    )
                    # Do convolution
                    conv3 = tf.nn.conv2d(
                        Z,
                        W_c3,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv3-%s" % unique_layer_id,
                    )
                    # Add in biases
                    conv_add_b3 = tf.nn.bias_add(conv3,
                                                 b_c3,
                                                 name="conv_add_b3-%s" %
                                                 unique_layer_id)
                    # Apply activation function on deep input
                    Z_final = tf.nn.relu(conv_add_b3,
                                         name="final_Z-%s" % unique_layer_id)
                Z_final = tf.reshape(Z_final, [-1, 1, 1, Z_shape[3]])
                all_pooled_outputs.append(Z_final)
                conv_layer_id += 1

        # Combine all the pooled activation maps into tensor
        self.n_filts_total = 0
        for (i, num) in enumerate(self.conv_p["deep_num_filts"]):
            if num == 0:
                self.n_filts_total += self.conv_p["num_filts"][i]
            else:
                self.n_filts_total += num
        # Concatenate all max activations to one vector
        Z_concat = tf.concat(all_pooled_outputs, axis=3)
        # Flatten activations to 2D tensor
        Z_flat = tf.reshape(Z_concat, [-1, self.n_filts_total], name="Z_flat")
        return Z_flat

    def fc_layers(self, inp):
        """
        Implements the fully connected layers.
        """
        with tf.device(self.next_device()), tf.name_scope('fc_layers'):
            layers = {}
            layer_compute = {}
            all_dims = [self.n_filts_total] + self.fc_p["num_filts"]
            for i in range(1, len(all_dims)):
                new_layer = {}
                # Define this FC layer's weights and biases
                new_layer["weights"] = tf.compat.v1.get_variable(
                    "W_fc_%s" % all_dims[i],
                    initializer=lambda: tf.random.truncated_normal(
                        [all_dims[i - 1], all_dims[i]], stddev=2e-1),
                    trainable=True,
                )
                self.W_for_l2.append(new_layer["weights"])
                new_layer["biases"] = tf.compat.v1.get_variable(
                    "b_fc_%s" % all_dims[i],
                    initializer=lambda: tf.random.truncated_normal(
                        [all_dims[i]], mean=0.5, stddev=1e-2),
                    trainable=True,
                )
                layers[i - 1] = new_layer

                # Multiply the inputs by weights
                xw_plus_b = tf.compat.v1.nn.xw_plus_b(
                    inp if i == 1 else layer_compute[i - 2],
                    layers[i - 1]["weights"],
                    layers[i - 1]["biases"],
                    name="xw_plus_b_%s" % all_dims[i],
                )
                # Apply activation
                fc_z = tf.math.sigmoid(xw_plus_b, name="fc_z_%s" % all_dims[i])
                # Perform dropout
                fc_z_dropped = tf.nn.dropout(
                    fc_z,
                    rate=1 - self.drop_keep_prob,
                    name="fc_z_dropped_%s" % all_dims[i],
                )
                layer_compute[i - 1] = tf.reshape(fc_z_dropped,
                                                  [-1, all_dims[i]])
            # Get variables for last FC layer
            if len(all_dims) == 1:
                last_input = inp
                last_dim = self.n_filts_total
            else:
                last_input = layer_compute[len(layer_compute) - 1]
                last_dim = all_dims[-1]
            return last_input, last_dim

    def U_project(self, inp, dim):
        """
        Implements the final fully-connected layer which maps the input to the
        output vector space, U.
        """
        # Define a reference to the feat vec of last FC layer before projection
        # onto U vector space.
        self.last_fc_z = tf.reshape(inp, [-1, dim], name="last_fc_z")
        # Define weights and biases for final projection
        W_fc = tf.compat.v1.get_variable(
            "last_W_fc",
            initializer=lambda: tf.random.truncated_normal([dim, self.out_dim],
                                                           stddev=2e-1),
            trainable=True,
        )
        # self.W_for_l2.append(W_fc)
        b_fc = tf.compat.v1.get_variable(
            "last_b_fc",
            initializer=lambda: tf.random.truncated_normal(
                [self.out_dim], mean=0.5, stddev=1e-2),
            trainable=True,
        )
        # Perform final projection to get U vectors
        U_vec = tf.compat.v1.nn.xw_plus_b(self.last_fc_z,
                                          W_fc,
                                          b_fc,
                                          name="U_vec")
        # Use sigmoid to squash vecs to (-1, 1) range
        U_vec = tf.math.sigmoid(U_vec, name="sig_U_vec")
        return U_vec

    def proto_init(self, rnd_noise_stddev=0.001):
        """
        Initialize the prototypes with the passed in initializations.
        """
        # Define the  prototypes
        return tf.Variable(
            initial_value=self.proto_inits + tf.random.truncated_normal(
                [utils.n_protos(),
                 utils.im_height(),
                 utils.im_width()],
                mean=0.0,
                stddev=rnd_noise_stddev,
            ),
            dtype=tf.float32,
            trainable=True)

    def rnd_proto_init(self):
        """
        Initialize the prototypes randomly.
        """
        # Define the prototypes (random init)
        return tf.Variable(
            initial_value=tf.random_uniform(
                [utils.n_protos(),
                 utils.im_height(),
                 utils.im_width()],
                minval=(-1 * self.rnd_proto_init_mag),
                maxval=self.rnd_proto_init_mag,
                seed=self.random_seed * 2,
            ),
            dtype=tf.float32,
            trainable=True)

    def _define_output_dirs(self):
        """
        Define and create if needed all the directories under the base
        directory that this class instatiation will be writing to.
        """
        # Create folder for all the results and models of this instance
        class_dir = os.path.join(self.base_save_path, self.name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Create output directory for batch, training and validation summaries
        self.batch_summary_path = os.path.join(
            class_dir, (self.name + "_batch_summary.csv"))
        self.train_summary_path = os.path.join(
            class_dir, (self.name + "_train_summary.csv"))
        self.val_summary_path = os.path.join(class_dir,
                                             (self.name + "_val_summary.csv"))
        print(("Writing batch summary to: %s" % self.batch_summary_path))
        print(("Writing training summary to: %s" % self.train_summary_path))
        print(("Writing val summary to: %s" % self.val_summary_path))
        self.batch_summary_file = open(self.batch_summary_path, "w")
        self.train_summary_file = open(self.train_summary_path, "w")
        self.val_summary_file = open(self.val_summary_path, "w")
        self.batch_summary_file.write(",".join(BATCH_SUMMARY_FORMAT) + "\n")
        self.train_summary_file.write(",".join(TRAIN_SUMMARY_FORMAT) + "\n")
        self.val_summary_file.write(",".join(VAL_SUMMARY_FORMAT) + "\n")

        # Create the path for the final summary
        self.final_summary_path = os.path.join(
            class_dir, (self.name + "_final_summary.csv"))
        print(("Writing final summary to: %s" % self.final_summary_path))
        self.final_summary_file = open(self.final_summary_path, "w")
        self.final_summary_file.write(",".join(FINAL_SUMMARY_FORMAT) + "\n")

        # Create the output directory for performance plots
        self.perf_plot_dir = os.path.join(class_dir, "perf_plots")
        if not os.path.exists(self.perf_plot_dir):
            os.makedirs(self.perf_plot_dir)
        print(("Writing performance plots to: %s" % self.perf_plot_dir))

        # Create the output directory for tsne plots
        self.tsne_plot_dir = os.path.join(class_dir, "tsne_plots")
        if not os.path.exists(self.tsne_plot_dir):
            os.makedirs(self.tsne_plot_dir)
        print(("Writing t-SNE plots to: %s" % self.tsne_plot_dir))

        # Create the output directory for prototype plots
        self.proto_plot_dir = os.path.join(class_dir, "proto_plots")
        if not os.path.exists(self.proto_plot_dir):
            os.makedirs(self.proto_plot_dir)
        print(("Writing prototype plots to: %s" % self.proto_plot_dir))

        # Create the output directory for models
        model_dir = os.path.join(class_dir, "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_fp = os.path.join(model_dir, self.name)
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        print(("Writing models to: %s\n" % self.model_fp))

    def _record_result(self, fp, result):
        """
        Take the result of a step of training or validation and record it in
        our summary writer, denoted fp.
        """
        str_results = []
        for r in result:
            if type(r) == list:
                str_results.append("|".join([str(e) for e in r]))
            else:
                str_results.append(str(r))
        fp.write(",".join(str_results) + "\n")
        fp.flush()

    def train(
        self,
        # A 3D batch of images with ndarray type (N, H, W).
        X,
        # The class labels in one-hot encoded format.
        y,
        # The probability of a neuron not being nullified in the dropout
        # layer during training.
        drop_keep_prob,
        # The size of batches to be used in training
        batch_size,
        # The optimization algorithm (Adam, Nadam or RMSprop).
        optimizer,
        # Number of batches in the training process until we run a
        # train/val set evaluation
        steps_per_eval,
        # Number of times we can run a val set evaluation without
        # improvement until we stop the training process
        max_evals_since_overwrite,
        # Alternatively, pass in an external validation set
        X_val=[],
        y_val=[],
        # Learning rate for the optimizer
        learning_rate=1e-3,
        # Set this to False to skip running evals on the training data if
        # you dont care about that. For instance, when doing hyperparameter
        # optimization.
        eval_on_train_data=True,
        # Set this to False to skip saving any batch results. This can end
        # up getting pretty large. It makes sense to set this to False if
        # you don't care about the history of batch loss, like if you are
        # doing hyperparameter optimization.
        record_batch_evals=True,
        # Instructions to plot every N steps. Set to 0 to avoid plotting.
        plot_every=0,
        plot_subset_size=0,
    ):

        self.plot_subset_size = plot_subset_size
        self.eval_on_train_data = eval_on_train_data
        with tf.Graph().as_default():
            session_conf = tf.compat.v1.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=LOG_DEVICE_PLACEMENT,
            )
            sess = tf.compat.v1.Session(config=session_conf)
            with sess.as_default():
                assert optimizer == "Adam" or optimizer == "SGD+M+N" or optimizer == "RMSprop"
                self._build_IUPG()  # Build the TF graph
                # Define Training procedure
                global_step = tf.Variable(0,
                                          name="global_step",
                                          trainable=False)
                optimizer_str = optimizer
                if optimizer == "Adam":
                    optimizer = tf.compat.v1.train.AdamOptimizer(
                        learning_rate=learning_rate)
                elif optimizer == "SGD+M+N":
                    optimizer = tf.compat.v1.train.MomentumOptimizer(
                        learning_rate=learning_rate,
                        momentum=0.9,
                        use_nesterov=True,
                    )
                elif optimizer == "RMSprop":
                    optimizer = tf.compat.v1.train.RMSPropOptimizer(
                        learning_rate=learning_rate)
                grads_and_vars = optimizer.compute_gradients(self.avg_loss_l2)
                train_op = optimizer.apply_gradients(grads_and_vars,
                                                     global_step=global_step)
                # Make sure all our output directories are set up
                self._define_output_dirs()
                # Initialize all variables
                sess.run(tf.compat.v1.global_variables_initializer())

                def train_step(batch, step_num):
                    """
                    Do a single training step. Record results.
                    """
                    feed_dict = {}
                    feed_dict[self.X_batch] = batch["X"]
                    feed_dict[self.y_batch] = batch["y"]
                    feed_dict[self.drop_keep_prob] = drop_keep_prob
                    _, cur_step_num, y_1d, loss, min_D, win_protos = sess.run(
                        [
                            train_op,
                            global_step,
                            self.y_batch_1d,
                            self.avg_loss,
                            self.min_D,
                            self.win_protos,
                        ],
                        feed_dict,
                    )
                    if cur_step_num == 10000:
                        self.sec_until_10000 = timer() - self.start
                        print(("\nSeconds until step 10000: %f\n" %
                               self.sec_until_10000))
                    # Get datetime string
                    time_str = datetime.now().strftime(DATETIME_FORMAT)
                    # Print and record results of batch
                    print(("{}: step {}, loss {:g}".format(
                        time_str, cur_step_num, loss)))
                    if record_batch_evals:
                        self._record_result(
                            self.batch_summary_file,
                            [time_str, cur_step_num, loss],
                        )

                def test_step(cur_step_num,
                              batch_iter,
                              record_fp=None,
                              bar=None):
                    """
                    Evaluates our current model on a val set. Records results.
                    Returns the loss and accuracy on the data.
                    """
                    all_losses = []
                    all_win_protos = []
                    all_min_D = []
                    all_y_true = []
                    pps = {}
                    for i in range(1, utils.n_protos() + 1):
                        pps[i] = {}
                        for j in range(utils.n_protos() + 1):
                            pps[i]["%d_dists" % j] = []
                    batch_num = 0
                    # Get current protos in U vector space
                    feed_dict = {
                        self.run_protos: True,
                        self.run_samps: False,
                        self.drop_keep_prob: 1.0,
                    }
                    U_protos = sess.run(self.proto_U_vec, feed_dict)
                    # Starting processing test input
                    for batch in batch_iter:
                        feed_dict = {}
                        feed_dict[self.X_batch] = batch["X"]
                        feed_dict[self.y_batch] = batch["y"]
                        feed_dict[self.drop_keep_prob] = 1.0
                        feed_dict[self.run_protos] = False
                        feed_dict[self.run_samps] = True
                        feed_dict[self.given_U_protos] = U_protos
                        # Calculate results
                        y_1d, losses, sig_D, min_D, win_protos = sess.run(
                            [
                                self.y_batch_1d,
                                self.losses,
                                self.sig_D,
                                self.min_D,
                                self.win_protos,
                            ],
                            feed_dict,
                        )
                        # Collect results
                        if type(losses) == list:
                            losses = losses[0]
                        all_losses.extend(losses.tolist())
                        all_y_true.extend(y_1d.tolist())
                        all_min_D.extend(min_D.tolist())
                        all_win_protos.extend(win_protos.tolist())
                        for i, y in enumerate(y_1d.tolist()):
                            for j, d in enumerate(sig_D[i, :].tolist(), 1):
                                pps[j]["%d_dists" % y].append(d)
                        batch_num += 1
                        # Update output
                        if bar is not None:
                            bar.next(len(batch["y"]))
                    if bar is not None:
                        bar.finish()

                    # Convert to numpy arrays
                    all_y_true = np.array(all_y_true)
                    all_min_D = np.array(all_min_D)
                    all_win_protos = np.array(all_win_protos)
                    # Combine all batch results into interpretable measures
                    result = {}
                    # Get average loss
                    result["avg_loss"] = np.mean(all_losses)
                    # Compute threshold which maximizes accuracy
                    opt_threshold = utils.get_opt_threshold(
                        all_y_true, all_min_D, all_win_protos)
                    all_y_pred = np.copy(all_win_protos)
                    all_y_pred[all_min_D >= opt_threshold] = 0
                    result["acc"] = np.mean(
                        (all_y_true == all_y_pred).astype("float"))

                    # Measure per proto stats
                    for i in range(1, utils.n_protos() + 1):
                        ps = pps[i]
                        for c in range(utils.n_protos() + 1):
                            if len(ps["%d_dists" % c]) > 0:
                                ps["avg_%d_dist" % c] = np.mean(ps["%d_dists" %
                                                                   c])
                                ps["std_%d_dist" % c] = np.std(ps["%d_dists" %
                                                                  c])
                                ps["max_%d_dist" % c] = np.max(ps["%d_dists" %
                                                                  c])
                                ps["min_%d_dist" % c] = np.min(ps["%d_dists" %
                                                                  c])
                                del pps[i]["%d_dists" % c]
                    result["pps"] = pps

                    # Print summarized results
                    time_str = datetime.now().strftime(DATETIME_FORMAT)
                    print((("{}: step {}, loss {:g}, acc {:g}\n").format(
                        time_str,
                        cur_step_num,
                        result["avg_loss"],
                        result["acc"],
                    )))

                    # Print confusion matrix and per proto stats
                    proto_labels = self.get_proto_labels()
                    result["cm"] = confusion_matrix(
                        all_y_true,
                        all_y_pred,
                        labels=[x for x in range(utils.n_classes_total())],
                    )
                    print("Confusion matrix:")
                    print_cm(result["cm"], proto_labels, normalize=False)
                    print("\nNormalized confusion matrix:")
                    print_cm(result["cm"], proto_labels)
                    print("\nMean per proto distances:")
                    print_pps(pps, "avg", proto_labels)
                    print("\nMax per proto distances:")
                    print_pps(pps, "max", proto_labels)
                    print("\nMin per proto distances:")
                    print_pps(pps, "min", proto_labels)

                    # Print number of samples per prototype
                    all_tar_win_protos = all_win_protos[
                        all_y_true > 0].tolist()
                    unique, counts = np.unique(all_tar_win_protos,
                                               return_counts=True)
                    all_counts = []
                    for i in range(1, utils.n_protos() + 1):
                        if i in unique:
                            all_counts.append(counts[unique == i][0])
                        else:
                            all_counts.append(0)
                    print("\nNumber of target samples per prototype:\n")
                    for i in range(len(all_counts)):
                        print(("\t%s: %d" % (proto_labels[i], all_counts[i])))
                    print("\n")

                    # Record results in summary file
                    if record_fp is not None:
                        self._record_result(
                            record_fp,
                            [
                                time_str,
                                cur_step_num,
                                result["avg_loss"],
                                result["acc"],
                                all_counts,
                            ],
                        )
                    return result

                # Feed input data into class instance to split and organize it
                iupg_input = IUPG_Input(
                    X,
                    y=y,
                    X_val=X_val,
                    y_val=y_val,
                    random_seed=self.random_seed,
                    n_classes=utils.n_classes_total(),
                )

                # Grab random subset of training data if we are going to plot
                if plot_every > 0:
                    random_sample = iupg_input.get_balanced_rnd_train_sample(
                        self.plot_subset_size)
                    self.plot_subset_size = len(random_sample["y"])
                # Keep track of the min val set loss (and associated stats)
                # we've seen so far
                min_val_loss = np.inf
                best_val_acc = -np.inf
                step_num_at_best = 0
                best_cm = None
                best_pps = None
                # The number of val set evaluations since we've beaten our
                # best val set loss. If this gets large, we can be confident
                # that we wont beat it in this training session.
                num_evals_since_overwrite = 0
                # Get a sense of performance before any training takes
                # place
                self.start = timer()
                if eval_on_train_data:
                    print("Doing initial train set evaluation...")
                    test_step(
                        0,
                        iupg_input.get_train_batch_iter(EVAL_BATCH_SIZE),
                        self.train_summary_file,
                        Bar("Progress... ", max=iupg_input.n_train),
                    )
                print("Doing initial val set evaluation...")
                test_step(
                    0,
                    iupg_input.get_val_batch_iter(EVAL_BATCH_SIZE),
                    self.val_summary_file,
                    Bar("Progress... ", max=iupg_input.n_val),
                )

                if plot_every > 0:
                    # Plot random subset at initialization
                    self.plot_U(
                        sess,
                        iupg_input.get_batch_iter(random_sample,
                                                  EVAL_BATCH_SIZE),
                        "0-%s" % self.name,
                        0,
                    )
                    self.plot_protos(sess, "0-%s" % self.name)
                # Enter the training loop
                print("\nBegninning training...\n")
                while num_evals_since_overwrite < max_evals_since_overwrite:
                    batch = iupg_input.next_batch(batch_size)
                    train_step(
                        batch,
                        tf.compat.v1.train.global_step(sess, global_step))
                    cur_step_num = tf.compat.v1.train.global_step(
                        sess, global_step)
                    # Check if plotting needs to be done
                    if (plot_every > 0) and (cur_step_num % plot_every == 0):
                        self.plot_U(
                            sess,
                            iupg_input.get_batch_iter(random_sample,
                                                      EVAL_BATCH_SIZE),
                            "%d-%s.png" % (cur_step_num, self.name),
                            cur_step_num,
                        )
                        self.plot_protos(
                            sess, "%d-%s.png" % (cur_step_num, self.name))
                    # Check if eval needs to be done
                    if cur_step_num % steps_per_eval == 0:
                        if eval_on_train_data:
                            print("\nDoing train set evaluation...")
                            train_batch_iter = iupg_input.get_train_batch_iter(
                                EVAL_BATCH_SIZE)
                            result = test_step(
                                cur_step_num,
                                train_batch_iter,
                                self.train_summary_file,
                                Bar("Progress... ", max=iupg_input.n_train),
                            )
                        print("\nDoing val set evaluation...")
                        val_batch_iter = iupg_input.get_val_batch_iter(
                            EVAL_BATCH_SIZE)
                        result = test_step(
                            cur_step_num,
                            val_batch_iter,
                            self.val_summary_file,
                            Bar("Progress... ", max=iupg_input.n_val),
                        )
                        # Check to see if a new best model has been found
                        if min_val_loss > result["avg_loss"]:
                            print((("New minimum loss found! Loss improved "
                                    "from: %f --> %f") %
                                   (min_val_loss, result["avg_loss"])))
                            path = self.saver.save(sess, self.model_fp)
                            print(("Saved new best model to %s\n" % path))
                            # Update best performance vars
                            min_val_loss = result["avg_loss"]
                            best_cm = result["cm"]
                            best_pps = result["pps"]
                            step_num_at_best = cur_step_num
                            feed_dict = {
                                self.run_protos: True,
                                self.run_samps: False,
                                self.drop_keep_prob: 1.0,
                            }
                            num_evals_since_overwrite = 0  # Reset this to 0
                        else:
                            print((("The loss was not improved. The current "
                                    "best loss is still: %f") % min_val_loss))
                            num_evals_since_overwrite += 1
                            print(("Number of evals since an overwrite: %d\n" %
                                   num_evals_since_overwrite))
                        if best_val_acc < result["acc"]:
                            best_val_acc = result["acc"]
                # Done with training
                print("The maximum number of evals per overwrite has been "
                      "exceeded. The final results are as follows:\n")
                # Print best results
                print(("Steps until best result was found: %d" %
                       step_num_at_best))
                print(("Best val loss: %f" % min_val_loss))
                # Define labels
                proto_labels = self.get_proto_labels()
                print("Best Val Confusion Matrix:")
                print_cm(best_cm, proto_labels, normalize=False)
                print("\nBest Val Confusion Matrix Normalized:")
                print_cm(best_cm, proto_labels)
                print("\nMean per proto distances:")
                print_pps(best_pps, "avg", proto_labels)
                print("\nMax per proto distances:")
                print_pps(best_pps, "max", proto_labels)
                print("\nMin per proto distances:")
                print_pps(best_pps, "min", proto_labels)

                self.best_cm = best_cm
                self.best_pps = best_pps

                # Write final summary
                final_summ_data = [
                    self.name,
                    batch_size,
                    drop_keep_prob,
                    steps_per_eval,
                    max_evals_since_overwrite,
                    optimizer_str,
                    self.l2_lambda,
                ]
                final_summ_data += [
                    self.conv_p["filt_sizes"],
                    self.conv_p["num_filts"],
                    self.conv_p["deep_filt_sizes"],
                    self.conv_p["deep_num_filts"],
                    self.conv_p["deep_pool"],
                ]
                if len(self.fc_p["num_filts"]) != 0:
                    final_summ_data += [self.fc_p["num_filts"]]
                else:
                    final_summ_data += [0]
                final_summ_data += [
                    learning_rate,
                    self.gamma,
                    utils.n_protos(),
                    self.out_dim,
                    self.dist_metric,
                    step_num_at_best,
                    min_val_loss,
                    best_val_acc,
                    self.sec_until_10000,
                ]
                self._record_result(self.final_summary_file, final_summ_data)
        print("\nFinished training...")
        self.train_summary_file.close()
        self.val_summary_file.close()
        self.final_summary_file.close()
        return {
            "loss": min_val_loss,
            "acc": best_val_acc,
            "best_cm": best_cm,
            "best_pps": best_pps,
        }

    def _read_summary_csv(self, fp):
        """
        Save all the information in a summary CSV into a dictionary.
        """
        data = {}
        with open(fp, "r") as csvfile:
            csv_reader = csv.reader(csvfile)
            headers = next(csv_reader)
            for h in headers:
                data[h] = []
            for row in csv_reader:
                for h, v in zip(headers, row):
                    if "STEP" in h:
                        if v == 0:
                            break  # Skip any pre-training analysis
                        data[h].append(int(v))
                    elif "LOSS" in h or "ACC" in h:
                        data[h].append(float(v))
                    elif "TIME" in h:
                        # Save time as seconds
                        datetime_object = datetime.strptime(v, DATETIME_FORMAT)
                        v = time.mktime(datetime_object.timetuple())
                        data[h].append(int(v))
        return data

    def graph_summaries(self):
        """
        Graph the results of training and validation which have been saved into
        CSV files on disk. Call this after training.
        """
        # Read in training and validation summary CSVs
        if self.eval_on_train_data:
            train_data_summary = self._read_summary_csv(
                self.train_summary_path)
        val_data_summary = self._read_summary_csv(self.val_summary_path)

        # Create loss plot
        plt.figure()
        if self.eval_on_train_data:
            plt.plot(
                train_data_summary["STEP"],
                train_data_summary["TRAIN_LOSS"],
                color="b",
                marker=".",
                linestyle="-",
                label="Training Set Loss",
            )
        plt.plot(
            val_data_summary["STEP"],
            val_data_summary["VAL_LOSS"],
            color="r",
            marker=".",
            linestyle="-",
            label="Testing Set Loss",
        )
        plt.xlabel("Batch Number")
        plt.ylabel("Average Loss")
        plt.xlim([1, max(val_data_summary["STEP"])])
        if self.eval_on_train_data:
            plt.ylim([
                0.0,
                max(
                    max(train_data_summary["TRAIN_LOSS"]),
                    max(val_data_summary["VAL_LOSS"]),
                ),
            ])
        else:
            plt.ylim([0.0, max(val_data_summary["VAL_LOSS"])])
        plt.grid()
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.perf_plot_dir, self.name) + "_loss.png")
        plt.close()

        # Create accuracy plot
        plt.figure()
        if self.eval_on_train_data:
            plt.plot(
                train_data_summary["STEP"],
                train_data_summary["TRAIN_ACC"],
                color="b",
                marker=".",
                linestyle="-",
                label="Training Set Accuracy",
            )
        plt.plot(
            val_data_summary["STEP"],
            val_data_summary["VAL_ACC"],
            color="r",
            marker=".",
            linestyle="-",
            label="Testing Set Accuracy",
        )
        plt.xlabel("Batch Number")
        plt.ylabel("Accuracy")
        plt.xlim([1, max(val_data_summary["STEP"])])
        if self.eval_on_train_data:
            plt.ylim([
                min(
                    min(train_data_summary["TRAIN_ACC"]),
                    min(val_data_summary["VAL_ACC"]),
                ),
                1.0,
            ])
        else:
            plt.ylim([min(val_data_summary["VAL_ACC"]), 1.0])
        plt.grid()
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.perf_plot_dir, self.name) + "_acc.png")
        plt.close()

    def plot_protos(self, sess, plot_name, v=True):
        """
        Create a plot to visualize the current state of the prototypes.
        """
        if v:
            print("Plotting prototypes...")
        # Get prototypes
        feed_dict = {
            self.run_protos: True,
            self.run_samps: False,
            self.drop_keep_prob: 1.0,
        }
        protos = sess.run(self.protos, feed_dict)
        self.plot_protos_helper(protos, plot_name)

    def plot_protos_helper(self, protos, plot_name):
        """
        Plot the passed in prototypes.
        """
        fig = plt.figure(figsize=(4.0, 10.0))
        grid = ImageGrid(fig, 111, nrows_ncols=(2, 5), axes_pad=0.0)
        for ax, proto in zip(grid, protos):
            # Iterating over the grid returns the Axes.
            proto = proto.astype("float")
            proto = proto - np.min(proto)
            proto = proto / np.max(proto)
            ax.imshow(proto, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
        plt.savefig(
            os.path.join(self.proto_plot_dir, self.name) + "-" + plot_name +
            ".png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_U(self, sess, batch_iter, plot_name, step_num, v=True):
        """
        Plot prototypes and a subset of data in the U space using t-SNE.
        """
        if v:
            print("Plotting data...")
        # Get prototypes
        feed_dict = {
            self.run_protos: True,
            self.run_samps: False,
            self.drop_keep_prob: 1.0,
        }
        U_protos = sess.run(self.proto_U_vec, feed_dict)

        # Get alpha vector
        alpha = sess.run(self.alpha, {})
        alpha = np.exp(alpha)

        # Get the sample feature vectors
        all_feat_vecs = []
        all_y = []
        if v:
            bar = Bar("--> Getting feature vectors... ",
                      max=self.plot_subset_size)
        for batch in batch_iter:
            feed_dict = {}
            feed_dict[self.X_batch] = batch["X"]
            feed_dict[self.y_batch] = batch["y"]
            feed_dict[self.drop_keep_prob] = 1.0
            feed_dict[self.run_protos] = False
            feed_dict[self.run_samps] = True
            feed_dict[self.given_U_protos] = U_protos

            y_1d, U_vecs = sess.run([self.y_batch_1d, self.samp_U_vec],
                                    feed_dict)

            all_feat_vecs.extend(U_vecs.tolist())
            all_y.extend(y_1d.tolist())

            if v:
                bar.next(len(batch["y"]))
        if v:
            bar.finish()

        all_feat_vecs = np.array(all_feat_vecs)
        n = len(all_feat_vecs)
        all_y = np.array(all_y)
        assert len(all_feat_vecs) == len(all_y)

        def siamese_dist(A, B, alpha):
            s = np.sum(alpha * np.abs(A - B))
            return (2.0 / (1.0 + np.exp(-2 * s))) - 1.0

        def l1_dist(A, B):
            s = np.sum(np.abs(A - B))
            return (2.0 / (1.0 + np.exp(-2 * s))) - 1.0

        if self.dist_metric == "siamese":
            metric = partial(siamese_dist, alpha=alpha)
        elif self.dist_metric == "abs_err":
            metric = l1_dist

        used_tsne = False
        if len(all_feat_vecs[0]) > 2 or self.dist_metric == "siamese":
            print("--> Performing t-SNE...")
            tsne = TSNE(
                n_components=2,
                init="pca",
                perplexity=30,
                verbose=0,
                metric=metric,
                random_state=self.random_seed,
                n_iter=5000,
            )
            break_point = len(all_feat_vecs)
            new_vecs = tsne.fit_transform(np.vstack((all_feat_vecs, U_protos)))
            all_feat_vecs = new_vecs[:break_point, :]
            U_protos = new_vecs[break_point:, :]
            used_tsne = True

        if v:
            print("--> Drawing plot...")
        data_by_y = [[] for j in range(utils.n_protos() + 1)]
        for i in range(n):
            data_by_y[int(all_y[i])].append(all_feat_vecs[i, :])
        for i in range(utils.n_protos() + 1):
            data_by_y[i] = np.asarray(data_by_y[i])

        colors = iter(plt.cm.hot(np.linspace(0.1, 0.75, utils.n_protos())))
        colors_by_proto = []

        plt.figure()
        ax = plt.subplot(111)

        # Plot benign data and malicious clusters
        for i in range(utils.n_protos() + 1):
            if i == 0:
                color = "b"
                label = "Noise"
            else:
                color = next(colors)
                colors_by_proto.append(color)
                label = "Prototype %d" % (i - 1)
            if self.int2label is not None:
                label = self.int2label[i]
            data = data_by_y[i]
            if len(data) > 0:
                ax.scatter(
                    x=data[:, 0],
                    y=data[:, 1],
                    color=color,
                    marker=".",
                    s=100,
                    alpha=0.4,
                    label=label,
                )
        # Plot prototypes
        for i in range(utils.n_protos()):
            ax.scatter(
                x=U_protos[i, 0],
                y=U_protos[i, 1],
                s=125,
                alpha=0.5,
                facecolors="k",
                edgecolors=colors_by_proto[i],
                label="Prototype %d" % i,
            )

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            fancybox=True,
            shadow=True,
            ncol=1,
        )

        ymin, ymax = ax.get_ylim()
        custom_ticks = np.linspace(ymin, ymax, 8)
        custom_tick_labels = ["%.2f" % tick for tick in custom_ticks]
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels(custom_tick_labels)

        xmin, xmax = ax.get_xlim()
        custom_ticks = np.linspace(xmin, xmax, 8)
        custom_tick_labels = ["%.2f" % tick for tick in custom_ticks]
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(custom_tick_labels)

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        if used_tsne:
            ax.set_title("t-SNE Visualization of U vector space")
        else:
            ax.set_title("IUPG Output Vector Space - Batch: %5d" %
                         int(step_num))
        plt.savefig(os.path.join(self.tsne_plot_dir, plot_name),
                    bbox_inches="tight")
        plt.close()

    def get_proto_labels(self):
        """
        Get labels for all prototypes. If int2label is not defined, the classes
        will have generic labels.
        """
        if self.int2label is None:
            proto_labels = ["Noise"] + [("Proto %d" % (i + 1))
                                        for i in range(utils.n_protos())]
        else:
            proto_labels = [
                self.int2label[i] for i in range(utils.n_classes_total())
            ]
        return proto_labels

    def plot_best_pps(self):
        """
        Plot the current state of the best_pps variable which holds the per
        proto stats at the snapshot of training with optimal loss.
        """
        # Define labels
        proto_labels = self.get_proto_labels()
        # Create dataframes
        for stat in ["min", "max", "avg", "std"]:
            df = []
            for proto_i in range(1, utils.n_protos() + 1):
                new_row = []
                for cls_i in range(utils.n_protos() + 1):
                    if "%s_%d_dist" % (stat, cls_i) in self.best_pps[proto_i]:
                        new_row.append(self.best_pps[proto_i]["%s_%d_dist" %
                                                              (stat, cls_i)])
                    else:
                        new_row.append(-1)
                df.append(new_row)
            df = np.array(df)
            df = pd.DataFrame(df, index=proto_labels[1:], columns=proto_labels)
            # Plot and save each fig
            save_fp = os.path.join(self.perf_plot_dir,
                                   "best_pps_%s.png" % stat)
            utils.pretty_plot_2darr(df, save_fp, arr_type="pps_%s" % stat)

    def plot_best_cm(self):
        """
        Plot the current state of the best_cm variable which holds the
        confusion matrix at the snapshot of training with optimal loss.
        """
        # Define labels
        proto_labels = self.get_proto_labels()
        # Create dataframe
        df_cm = pd.DataFrame(self.best_cm,
                             index=proto_labels,
                             columns=proto_labels)
        # Define save path
        save_fp = os.path.join(self.perf_plot_dir, "best_cm.png")
        # Send to plotting
        utils.pretty_plot_2darr(df_cm, save_fp)
