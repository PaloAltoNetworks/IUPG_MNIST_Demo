"""
utils.py

Various utilities for the IUPG_MNIST_Demo

@author: Brody Kutt (bkutt@paloaltonetworks.com)
Copyright (c) 2020 Palo Alto Networks
"""

import os
import csv
import matplotlib

matplotlib.use("Agg")  # Force no use of Xwindows backend
import numpy as np
import seaborn as sn
from progress.bar import Bar
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh


def random_seed():
    """
    For repeatability.
    """
    return 1994


def n_digits():
    """
    Number of unique digits.
    """
    return 10


def n_protos():
    """
    Number of prototypes (1 for each class in this demo).
    """
    return 10


def n_target_classes():
    """
    Number of target classes (that have >= 1 prototype assigned to them).
    """
    return 10


def n_classes_total():
    """
    Number of target and off-target classes.
    """
    return n_target_classes() + 1


def im_height():
    """
    Height of each image.
    """
    return 28


def im_width():
    """
    Width of each image.
    """
    return 28


def datetime_str():
    """
    Get a string that displays the current date and time in standard format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def iupg_config_base():
    """
    Returns the base path where all the IUPGconfig files are stored.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(cur_dir, "configs")


def load_npz_data(path, load_y=True):
    """
    Load all contents of a dataset from an npz compressed archive.
    """
    result = {}
    data = None
    try:
        data = np.load(path, mmap_mode=None, allow_pickle=True)
    except IOError:
        pass
    if data is None:
        try:
            path += ".npz"  # Try adding the file extension
            data = np.load(path, mmap_mode=None, allow_pickle=True)
        except IOError:
            return None
    result["X"] = data["X"]
    if load_y:
        result["y"] = data["y"]
        result["int2label"] = data["int2label"].item()
    return result


def record_scores_and_preds(dig_ids,
                            all_sig_D,
                            all_y_pred,
                            save_fp,
                            all_y_true=None):
    """
    Take in results and save them to disk in a CSV file.
    """
    with open(save_fp, "w") as csv_file:
        writer = csv.writer(csv_file)
        # Create header row
        header_row = ["DIGIT_ID"]
        for i in range(all_sig_D.shape[-1]):
            header_row += ["%d_SCORE" % i]
        header_row += ["Y_PRED"]
        if all_y_true is not None:
            header_row += ["Y_TRUE"]
        writer.writerow(header_row)
        # Create data rows
        for i in range(len(dig_ids)):
            new_row = [dig_ids[i]]
            sig_D = all_sig_D[i, :]
            new_row += [float(j) for j in sig_D]
            new_row += [all_y_pred[i]]
            if all_y_true is not None:
                new_row += [all_y_true[i]]
            writer.writerow(new_row)


def get_opt_threshold(y_true, min_D, win_protos):
    """
    Calculate the decision threshold (to call noise samples) which maximizes
    accuracy.
    """
    min_D_inds = np.argsort(min_D)
    sorted_y_true = y_true[min_D_inds]
    sorted_min_D = min_D[min_D_inds]
    sorted_win_protos = win_protos[min_D_inds]
    n_samps = len(sorted_y_true)
    candidate_thresholds = [sorted_min_D[0] / 2.0] + ((sorted_min_D[1:] + sorted_min_D[:-1]) / 2.0).tolist() + [(1.0 + sorted_min_D[-1]) / 2.0] + [2.0]
    candidate_thresholds = np.array(list(set(candidate_thresholds)))
    n_threshs = len(candidate_thresholds)
    min_D_tiled = np.tile(sorted_min_D, (n_threshs, 1))
    win_protos_tiled = np.tile(sorted_win_protos, (n_threshs, 1))
    y_true_tiled = np.tile(sorted_y_true, (n_threshs, 1))
    y_pred_tiled = (min_D_tiled < np.reshape(
        candidate_thresholds,
        (n_threshs, 1))).astype("float") * win_protos_tiled
    corr_preds = (y_true_tiled == y_pred_tiled).astype("int")
    accs = np.sum(corr_preds, axis=1) / float(n_samps)
    arg_max_acc = np.argmax(accs)
    return candidate_thresholds[arg_max_acc]


def print_iupg_params(params):
    """
    Print the hyperparameters of IUPG.
    """
    print(("U Vec Size: %d" % params["out_dim"]))
    print(("Distance Metric: %s" % params["dist_metric"]))
    print(("Batch size: %d" % params["batch_size"]))
    print(("Maximum evaluations since an overwrite: %d" %
           params["max_evals_since_overwrite"]))
    print(("Steps until an evaluation: %d" % params["steps_per_eval"]))
    print(("Optimization algorithm: %s" % params["optimizer"]))
    print(("Gamma: %f" % params["gamma"]))
    print("CNN params:")
    c = params["cnn_params"]
    print(("\tFilter sizes: %s" % str(c["filt_sizes"])))
    print(("\tNumber of filters per layer are: %s" % str(c["num_filts"])))
    print(("\tDeep filter sizes: %s" % str(c["deep_filt_sizes"])))
    print(
        ("\tNumber of deep filters per layer: %s" % str(c["deep_num_filts"])))
    print(("\tChoices of deep pooling: %s" % str(c["deep_pool"])))
    print("FC params:")
    print(("\tNumber of filters per layer are: %s" %
           str(params["fc_params"]["num_filts"])))
    print(("Dropout keep probability: %f" % params["drop_keep_prob"]))
    print(("Learning rate: %f" % params["learning_rate"]))
    print(("L2 lambda: %f" % params["l2_lambda"]))
    if "rnd_proto_init_mag" in params:
        print(("Random Proto Init Mag: %f" % params["rnd_proto_init_mag"]))
    print(("Random seed: %d\n" % params["random_seed"]))


def configcell_text_and_colors(
    array_df,
    arr_type,
    lin,
    col,
    oText,
    facecolors,
    posi,
    fz,
    fmt,
    show_null_values=0,
):
    """
    config cell text and colors and return text elements to add and to del
    """
    text_add = []
    text_del = []
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    if tot_all > 0:
        per = (float(cell_val) / tot_all) * 100
    else:
        per = 0.0
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # Last line and/or last column
    if arr_type == "cm" and ((col == (ccl - 1)) or (lin == (ccl - 1))):
        # Totals and percents
        if cell_val != 0:
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif col == ccl - 1:
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif lin == ccl - 1:
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ["%.2f%%" % (per_ok), "100%"][per_ok == 100]

        # Text to DEL
        text_del.append(oText)

        # Text to ADD
        font_prop = fm.FontProperties(weight="bold", size=fz)
        text_kwargs = dict(
            color="w",
            ha="center",
            va="center",
            gid="sum",
            fontproperties=font_prop,
        )
        lis_txt = ["%d" % (cell_val), per_ok_s, "%.2f%%" % (per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy()
        dic["color"] = "g"
        lis_kwa.append(dic)
        dic = text_kwargs.copy()
        dic["color"] = "r"
        lis_kwa.append(dic)
        lis_pos = [
            (oText._x, oText._y - 0.3),
            (oText._x, oText._y),
            (oText._x, oText._y + 0.3),
        ]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0],
                           y=lis_pos[i][1],
                           text=lis_txt[i],
                           kw=lis_kwa[i])
            text_add.append(newText)

        # Set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr
    else:
        if arr_type == "cm":
            if per > 0:
                txt = "%s\n%.2f%%" % (cell_val, per)
            else:
                if show_null_values == 0:
                    txt = ""
                elif show_null_values == 1:
                    txt = "0"
                else:
                    txt = "0\n0.0%"
        else:
            if per > 0:
                txt = "%.4f" % (cell_val)
            else:
                if show_null_values == 0:
                    txt = ""
                else:
                    txt = "0.0"
        oText.set_text(txt)

        # Main diagonal
        if (arr_type == "cm" and col == lin) or (arr_type != "cm" and
                                                 (col - 1) == lin):
            # Set color of the textin the diagonal to white
            oText.set_color("w")
            # Set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color("r")
    return text_add, text_del


def pretty_plot_2darr(
    df_cm,
    save_fp,
    annot=True,
    cmap="Oranges",
    fmt=".2f",
    fz=11,
    lw=0.5,
    cbar=True,
    figsize=[12, 10],
    show_null_values=1,
    arr_type="cm",
):
    """
    print a 2D array with default layout (like matlab)
    params:
      df_cm          dataframe (pandas) without totals
      annot          print text in each cell
      cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
      fz             fontsize
      lw             linewidth
      arr_type       'cm' for plotting a confusion matrix
                     'pps_avg' for plotting per-proto avg distances
                     'pps_min' for plotting per-proto min distances
                     'pps_max' for plotting per-proto max distances
                     'pps_std' for plotting per-proto std distances
    """
    if arr_type == "cm":
        # Create "Total" row and column
        sum_col = []
        for c in df_cm.columns:
            sum_col.append(df_cm[c].sum())
        sum_lin = []
        for item_line in df_cm.iterrows():
            sum_lin.append(item_line[1].sum())
        df_cm["sum_lin"] = sum_lin
        sum_col.append(np.sum(sum_lin))
        df_cm.loc["sum_col"] = sum_col

    # To print always in the same window
    fig1 = plt.figure("2darr plotting default", figsize)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # Clear existing plot

    # Thanks for seaborn
    if arr_type == "cm":
        ax = sn.heatmap(
            df_cm,
            annot=annot,
            annot_kws={"size": fz},
            linewidths=lw,
            ax=ax1,
            cbar=cbar,
            cmap=cmap,
            linecolor="w",
            fmt=fmt,
        )
    else:
        ax = sn.heatmap(
            df_cm,
            annot=annot,
            annot_kws={"size": fz},
            linewidths=lw,
            ax=ax1,
            cbar=cbar,
            cmap=cmap + "_r",
            linecolor="w",
            fmt=fmt,
            vmin=0.0,
            vmax=1.0,
        )

    # Set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)
    for t in ax.yaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)

    # Face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # Iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = []
    text_del = []
    posi = -1  # From left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1])
        col = int(pos[0])
        posi += 1
        # Set text
        txt_res = configcell_text_and_colors(
            array_df,
            arr_type,
            lin,
            col,
            t,
            facecolors,
            posi,
            fz,
            fmt,
            show_null_values=show_null_values,
        )

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # Remove the old ones
    for item in text_del:
        item.remove()
    # Append the new ones
    for item in text_add:
        ax.text(item["x"], item["y"], item["text"], **item["kw"])

    # Titles and legends
    if arr_type == "cm":
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    else:
        ax.set_xlabel("Data Label")
        ax.set_ylabel("Prototype")
    if arr_type == "pps_avg":
        ax.set_title("Mean of Prototype to Labeled Data Distances")
    elif arr_type == "pps_min":
        ax.set_title("Minimum of Prototype to Labeled Data Distances")
    elif arr_type == "pps_max":
        ax.set_title("Maximum of Prototype to Labeled Data Distances")
    elif arr_type == "pps_std":
        ax.set_title(
            "Standard Deviation of Prototype to Labeled Data Distances")
    plt.tight_layout()  # Set layout slim
    plt.savefig(save_fp)
    plt.close()
