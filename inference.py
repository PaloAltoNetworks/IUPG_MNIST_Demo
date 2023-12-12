"""
IUPG_train.py

Convienience script to run inference of IUPG instance.

@author: Brody Kutt (bkutt@paloaltonetworks.com)
Copyright (c) 2020 Palo Alto Networks
"""

import os
import utils
import argparse
import numpy as np
from IUPG_Evaluator import IUPG_Evaluator

if __name__ == "__main__":
    # Parse the arguments and options
    parser = argparse.ArgumentParser(
        description=("Use a IUPG model to perform inference."))
    parser.add_argument(
        "--model_dir",
        required=True,
        help=("The directory where your model lives."),
    )
    parser.add_argument(
        "--cust_thresh",
        required=False,
        default=-1.0,
        type=float,
        help=("Pass in a threshold to call noise. Otherwise, compute the "
              "optimal threshold."),
    )
    parser.add_argument(
        "--save_fp",
        required=True,
        help=("The filepath you want to save all results in."),
    )
    parser.add_argument("--npz_fp",
                        required=True,
                        help=("The NPZ file containing the data."))
    parser.add_argument(
        "--gpu_id",
        required=False,
        default=None,
        help=("Specificy a GPU to use by its ID (default: None)."),
    )
    args = parser.parse_args()
    if not os.path.isdir(os.path.dirname(args.save_fp)):
        os.makedirs(os.path.dirname(args.save_fp))

    if args.gpu_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    print(("-" * 80))
    print("Loading model...")
    iupg = IUPG_Evaluator(args.model_dir)
    print(("--> Loaded model at: %s" % args.model_dir))

    print("Loading the data...")
    data = utils.load_npz_data(args.npz_fp)
    print(("--> Loaded %d samples" % len(data["y"])))

    print("Making predictions...")
    all_sig_D = iupg.predict(X=data["X"], v=True)
    all_min_D = np.min(all_sig_D, axis=1)
    all_win_protos = np.argmin(all_sig_D, axis=1) + 1

    if args.cust_thresh < 0:
        # Get optimal threshold
        print('--> Getting optimal threshold...')
        thresh = utils.get_opt_threshold(data["y"], all_min_D, all_win_protos)
        print(('----> Optimal threshold computed: %f' % thresh))
    else:
        print(('--> Using custom threshold: %f' % args.cust_thresh))
        thresh = args.cust_thresh
    # Call relevant samples noise
    all_y_pred = np.copy(all_win_protos)
    all_y_pred[all_min_D >= thresh] = 0

    print("Writing prediction CSV...")
    utils.record_scores_and_preds(
        [str(i) for i in range(len(data["y"]))],
        all_sig_D,
        all_y_pred,
        args.save_fp,
        all_y_true=data["y"],
    )
    print(("--> Wrote predictions to: %s" % args.save_fp))
    print("\nExiting...")
    print(("-" * 80))
