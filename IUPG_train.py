"""
IUPG_train.py

Convienience script to facilitate the training of an IUPG instance.

@author: Brody Kutt (bkutt@paloaltonetworks.com)
Copyright (c) 2020 Palo Alto Networks
"""

import os
import sys
import utils
import random
import argparse
import matplotlib
matplotlib.use("Agg")  # Force no use of Xwindows backend
import numpy as np
import configparser
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from IUPG_Builder import IUPG_Builder
from scipy.spatial.distance import cdist


def process_config(config):
    """
    Take a loaded config instance and transfer it into a dictionary with the
    correct datatypes.
    """
    args = {}
    # Grab critical params
    args["X_fp"] = config.get("Critical", "X_fp")
    args["X_val_fp"] = config.get("Critical", "X_val_fp")
    args["save_dir"] = config.get("Critical", "save_dir")
    if not os.path.isdir(args["save_dir"]):
        os.makedirs(args["save_dir"])
    args["model_name"] = os.path.basename(args["save_dir"])
    args["out_dim"] = config.getint("Critical", "out_dim")
    args["dist_metric"] = config.get("Critical", "dist_metric")
    args["drop_keep_prob"] = config.getfloat("Critical", "drop_keep_prob")
    args["batch_size"] = config.getint("Critical", "batch_size")
    args["steps_per_eval"] = config.getint("Critical", "steps_per_eval")
    args["max_evals_since_overwrite"] = config.getint(
        "Critical", "max_evals_since_overwrite")
    args["optimizer"] = config.get("Critical", "optimizer")
    args["gamma"] = config.getfloat("Critical", "gamma")

    args["conv_params"] = {}
    # Grab Char CNN params
    c = args["conv_params"]
    c["filt_sizes"] = config.get("Conv_Params", "filt_sizes")
    c["filt_sizes"] = [int(i.strip()) for i in c["filt_sizes"].split(",")]
    c["num_filts"] = config.get("Conv_Params", "num_filts")
    c["num_filts"] = [int(i.strip()) for i in c["num_filts"].split(",")]
    c["deep_filt_sizes"] = config.get("Conv_Params", "deep_filt_sizes")
    c["deep_filt_sizes"] = [
        int(i.strip()) for i in c["deep_filt_sizes"].split(",")
    ]
    c["deep_num_filts"] = config.get("Conv_Params", "deep_num_filts")
    c["deep_num_filts"] = [
        int(n.strip()) for n in c["deep_num_filts"].split(",")
    ]
    c["deep_pool"] = config.get("Conv_Params", "deep_pool")
    c["deep_pool"] = [
        i.strip().startswith("T") for i in c["deep_pool"].split(",")
    ]
    # Grab fully connected layer params
    args["fc_params"] = {}
    args["fc_params"]["num_filts"] = config.get("FC_Params", "num_filts")
    if args["fc_params"]["num_filts"] == "":
        args["fc_params"]["num_filts"] = []
    else:
        args["fc_params"]["num_filts"] = [
            int(i.strip()) for i in args["fc_params"]["num_filts"].split(",")
        ]
    # Grab optional params
    try:
        args["l2_lambda"] = config.getfloat("Optional", "l2_lambda")
    except configparser.NoOptionError:
        args["l2_lambda"] = 0.0
    try:
        args["learning_rate"] = config.getfloat("Optional", "learning_rate")
    except configparser.NoOptionError:
        args["learning_rate"] = 1e-3
    try:
        args["random_seed"] = config.getint("Optional", "random_seed")
    except configparser.NoOptionError:
        args["random_seed"] = 1994
    random.seed(args["random_seed"])
    np.random.seed(args["random_seed"])
    try:
        args["eval_on_train_data"] = config.get(
            "Optional", "eval_on_train_data").strip().startswith("T")
    except configparser.NoOptionError:
        args["eval_on_train_data"] = True
    try:
        args["record_batch_evals"] = config.get(
            "Optional", "record_batch_evals").strip().startswith("T")
    except configparser.NoOptionError:
        args["record_batch_evals"] = True
    try:
        args["plot_every"] = config.getint("Optional", "plot_every")
    except configparser.NoOptionError:
        args["plot_every"] = 0
    try:
        args["plot_subset_size"] = config.getint("Optional",
                                                 "plot_subset_size")
    except configparser.NoOptionError:
        args["plot_subset_size"] = 0
    try:
        args["use_k_means"] = config.get("Optional",
                                         "use_k_means").strip().startswith("T")
    except configparser.NoOptionError:
        args["use_k_means"] = False
    try:
        args["global_max_pooling"] = config.get(
            "Optional", "global_max_pooling").strip().startswith("T")
    except configparser.NoOptionError:
        args["global_max_pooling"] = True
    try:
        args["rnd_proto_init_mag"] = config.getfloat("Optional",
                                                     "rnd_proto_init_mag")
    except configparser.NoOptionError:
        args["rnd_proto_init_mag"] = 1.0
    return args


def train_iupg(args, proto_inits=None):
    """
    Train IUPG with the given parameters. See parameter descriptions for the
    IUPG specific parameters in IUPG_Builder.py.
    """
    iupg = IUPG_Builder(
        args["model_name"],
        os.path.dirname(args["save_dir"]),
        args["conv_params"],
        args["fc_params"],
        args["out_dim"],
        args["dist_metric"],
        proto_inits,
        args["rnd_proto_init_mag"],
        args["l2_lambda"],
        args["gamma"],
        args["random_seed"],
        args["int2label"],
        args["global_max_pooling"],
    )
    iupg.train(
        args["X"],
        args["y"],
        args["drop_keep_prob"],
        args["batch_size"],
        args["optimizer"],
        args["steps_per_eval"],
        args["max_evals_since_overwrite"],
        args["X_val"],
        args["y_val"],
        args["learning_rate"],
        args["eval_on_train_data"],
        args["record_batch_evals"],
        args["plot_every"],
        args["plot_subset_size"],
    )
    print("Graphing summaries...")
    iupg.graph_summaries()
    print("Plotting best confusion matrix...")
    iupg.plot_best_cm()
    print("Plotting best per proto stats...")
    iupg.plot_best_pps()


if __name__ == "__main__":
    # Parse the arguments and options
    parser = argparse.ArgumentParser(description=("Train an IUPG model."))
    parser.add_argument(
        "--config_files",
        nargs="+",
        required=True,
        help=("The list of configuration files. Each configuration will be"
              "executed in sequence in the order that they were passed in."),
    )
    parser.add_argument(
        "--gpu_id",
        required=False,
        default=None,
        help=("Specificy a GPU to use by its ID (default: None)."),
    )
    args = parser.parse_args()

    if args.gpu_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Search for the config files
    configs = [configparser.ConfigParser() for c in args.config_files]
    for (i, config) in enumerate(configs):
        full_path = os.path.join(utils.iupg_config_base(),
                                 args.config_files[i])
        if config.read(full_path) == []:
            full_path += ".ini"  # Try adding the file extension
            if config.read(full_path) == []:
                print((('ERR: IUPG_train.py: Config file "%s" is missing or '
                        "empty.") % full_path))
                sys.exit(-1)  # Exit failure
    # Process the config files
    arg_sets = [process_config(config) for config in configs]
    old_args = None
    for (i, args) in enumerate(arg_sets):
        print(("-" * 80))
        print(("Processing config #%d of %d...\n" % (i + 1, len(arg_sets))))
        print("Loading the training data...")
        result = utils.load_npz_data(args["X_fp"])
        print(("--> Loaded %d samples" % len(result["y"])))
        args["X"] = result["X"]
        args["y"] = result["y"]
        args["int2label"] = result["int2label"]

        # Make sure training data is properly shuffled
        n = len(args["y"])
        new_order = random.sample(list(range(n)), n)
        args["X"] = args["X"][new_order]
        args["y"] = args["y"][new_order]

        print("Loading the validation data...")
        result = utils.load_npz_data(args["X_val_fp"])
        print(("--> Loaded %d samples" % len(result["y"])))
        args["X_val"] = result["X"]
        args["y_val"] = result["y"]

        # Define proto inits, if required
        proto_inits = None
        if args["use_k_means"]:
            print("Performing clustering to determine proto inits...")
            proto_inits = np.zeros(
                (utils.n_digits(), utils.im_height(), utils.im_width()))
            if not os.path.isdir(os.path.join(args["save_dir"],
                                              "kmeans_plots")):
                os.mkdir(os.path.join(args["save_dir"], "kmeans_plots"))
            for digit_id in range(utils.n_digits()):
                print(("--> Working on digit %d..." % digit_id))
                y_id = digit_id + 1

                print("----> Isolating valid images...")
                iso_X = args["X"][args["y"] == y_id]
                iso_y = args["y"][args["y"] == y_id]
                flat_images = iso_X.reshape((len(iso_X), -1))

                print("----> Performing PCA...")
                reduced_data = PCA(n_components=min(
                    75, flat_images.shape[0])).fit_transform(flat_images)
                print("----> Performing K-Means...")
                kmeans = KMeans(
                    init="k-means++",
                    n_clusters=1,
                    n_init=10,
                    random_state=args["random_seed"],
                )
                kmeans.fit(reduced_data)
                print(("------> Discovered cluster center of shape: %s" %
                       str(kmeans.cluster_centers_.shape)))
                print("----> Calculating all distances...")
                dists = cdist(kmeans.cluster_centers_, reduced_data,
                              "euclidean")
                print(("------> Minimum distance: %f" % np.min(dists, axis=1)))
                print(("------> Mean distance: %f" % np.mean(dists, axis=1)))
                argmin = np.argmin(dists, axis=1)
                print("----> Grabbing corresponding digits...")
                proto_inits[digit_id, :, :] = flat_images[argmin].reshape(
                    (1, utils.im_height(), utils.im_width()))
                fig, ax = plt.subplots()
                ax.imshow(proto_inits[digit_id])
                plt.savefig(
                    os.path.join(
                        args["save_dir"],
                        "kmeans_plots",
                        "digit_%d_init.png" % digit_id,
                    ))
                plt.show()
                plt.close()

        # If data is loaded, start the training
        print("Training IUPG with the following settings:\n")
        print(("Number of training samples is: %s" % str(args["y"].shape)))
        print(
            ("Number of validation samples is: %s" % str(args["y_val"].shape)))
        utils.print_iupg_params(args)
        train_iupg(args, proto_inits=proto_inits)
    print("\nExiting...")
    print(("-" * 80))
