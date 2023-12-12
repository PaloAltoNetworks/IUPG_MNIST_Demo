"""
analyze_scores.py

Analyze the output of inference.py

@author: Brody Kutt (bkutt@paloaltonetworks.com)
Copyright (c) 2020 Palo Alto Networks
"""

import os
import sys
import csv
import utils
import argparse
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score


def floatToString(inp):
    """
    Print a float nicely.
    """
    return ("%.15f" % inp).rstrip("0").rstrip(".")


def read_pred_file(path):
    """
    Read in the output of inference.py and load everything into numpy arrays.
    """
    data = {}
    with open(path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = next(csv_reader)
        for h in headers:
            data[h] = []
        n_rows = 0
        for row in csv_reader:
            for h, v in zip(headers, row):
                if "DIGIT_ID" in h:
                    data[h].append(str(v))
                elif "SCORE" in h:
                    data[h].append(float(v))
                elif "Y_PRED" in h or "Y_TRUE" in h:
                    data[h].append(int(v))
            n_rows += 1
    all_sig_D = np.zeros((n_rows, utils.n_protos()))
    for i in range(utils.n_protos()):
        all_sig_D[:, i] = data["%d_SCORE" % i]
    data["all_sig_D"] = all_sig_D
    data["all_min_D"] = np.min(all_sig_D, axis=1)
    data["y_true_np"] = np.array(data["Y_TRUE"])
    data["y_pred_np"] = np.array(data["Y_PRED"])
    return data


if __name__ == "__main__":
    # Parse the arguments and options
    parser = argparse.ArgumentParser(
        description="Do analysis on computed malicious class scores.")
    parser.add_argument(
        "--pred_fps",
        nargs="+",
        help=("All filepaths leading to prediction files you wish compare. "
              "Separate each with a space."),
    )
    parser.add_argument(
        "--labels",
        help=("Labels for prediction files. Supply one for each prediction "
              "file. Separate each with a comma."),
    )
    parser.add_argument(
        "--cust_thresh",
        required=False,
        default=-1.0,
        type=float,
        help=("Pass in a threshold to call noise. Otherwise, use the results "
              "already present in Y_PRED."),
    )
    parser.add_argument(
        "--ref_fprs",
        default="",
        required=False,
        metavar="fpr1,fpr2,...",
        help=("All FPRs which you want to discover corresponding recall. "
              "Separate each with a comma."),
    )

    args = parser.parse_args()
    labels = args.labels.strip().split(",")
    ref_fprs = [float(i) for i in args.ref_fprs.strip().split(",") if i != ""]

    for fp in args.pred_fps:
        assert os.path.isfile(fp)

    print(("-" * 80))
    print("Reading in predictions files...")
    all_data = []
    for i, fp in enumerate(args.pred_fps):
        all_data.append(read_pred_file(fp))
        print(("\tRead in %s predictions!" % labels[i]))

    print("\nComputing accuracies...\n")
    all_accuracy = []
    for i, data in enumerate(all_data):
        print(("\tUsing predictions with label '%s':" % labels[i]))
        if args.cust_thresh < 0:
            print('--> Using results in Y_PRED')
            acc = accuracy_score(data["y_true_np"], data["y_pred_np"])
        else:
            print(('--> Using custom threshold: %f' % args.cust_thresh))
            y_pred = np.argmin(data["all_sig_D"], axis=1) + 1
            y_pred[data["all_min_D"] >= args.cust_thresh] = 0
            acc = accuracy_score(data["y_true_np"], y_pred)
        print(("\t\tAccuracy: %f" % acc))
        all_accuracy.append(acc)

    print("\nAccuracies in copy/paste friendly format...\n")
    print((",".join(labels)))
    print((",".join([str(a) for a in all_accuracy])))

    if len(ref_fprs) > 0:
        print("\nComputing ROC curves...")
        if np.min(data["y_true_np"]) != 0:
            print("--> Your data doesn't contain any noise samples.")
            print("\nExiting...")
            print(("-" * 80))
            sys.exit()
        all_fprs, all_threshs = [], []
        for i, data in enumerate(all_data):
            print(("\n\tUsing predictions with label '%s':" % labels[i]))
            data["all_min_D"] = np.amin(data["all_sig_D"], axis=1)
            all_min_D_inv = 1.0 - data["all_min_D"]

            y_true_binary = (data["y_true_np"] > 0).astype("int")
            fpr, _, threshs = roc_curve(y_true_binary, all_min_D_inv)
            all_fprs.append(fpr)
            all_threshs.append(threshs)

        all_nn_errs = {}
        for ref_fpr in ref_fprs:
            all_nn_errs[str(ref_fpr)] = []

        print("\nComputing stats at configured thresholds...")
        for i, data in enumerate(all_data):
            print(("\n\tUsing predictions with label '%s':" % labels[i]))
            for ref_fpr in ref_fprs:
                print(("\t\tStats using reference FPR <= %s..." %
                       floatToString(ref_fpr)))
                idx = np.sum(all_fprs[i] <= ref_fpr) - 1
                fpr = all_fprs[i][idx]
                opt_thresh = 1.0 - all_threshs[i][idx]
                print(("\t\t\tConfigured Threshold: %.10f, FPR: %.10f" %
                       (opt_thresh, fpr)))
                # Produce predictions with opt threshold
                y_preds = []
                for j in range(len(data["Y_TRUE"])):
                    sig_D = data["all_sig_D"][j, :]
                    min_D = data["all_min_D"][j]
                    if min_D <= opt_thresh:
                        y_preds.append(np.argmin(sig_D) + 1)
                    else:
                        y_preds.append(0)
                y_preds = np.array(y_preds)

                # Produce compute cumulative recall and FPR
                nn_y_true = data["y_true_np"][data["y_true_np"] != 0]
                nn_y_pred = y_preds[data["y_true_np"] != 0]
                nn_err = 1.0 - accuracy_score(
                    nn_y_true, nn_y_pred, normalize=True)
                all_nn_errs[str(ref_fpr)].append(nn_err)

                n_y_true = data["y_true_np"][data["y_true_np"] == 0]
                n_y_pred = y_preds[data["y_true_np"] == 0]
                new_fpr = np.sum(
                    (n_y_pred > 0).astype("int")) / float(len(n_y_true))
                print(("\t\t\tNon-Noise Error: %.10f, New FPR: %.10f" %
                       (nn_err, new_fpr)))

    print("\nExiting...")
    print(("-" * 80))
