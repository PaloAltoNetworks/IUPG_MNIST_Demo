# IUPG_MNIST_Demo

A simple demo implementation of the Innocent Until Proven Guilty (IUPG) learning framework to train an MNIST classifier with or without noise.

This demo is built to train and test an IUPG network with <em>one</em> designated prototype per target class. In this case, each digit 0-9 is defined as a target class. One off-target class (class label 0) is also supported. Two different sources of noise can be tested with this demo: Gaussian noise and random strokes. K-Means++ clustering on the training set can be used to discover useful prototype intializations for each class. If clustering is not used, the prototypes will be initialized randomly.

This demo does not currently implement the strategy of using a basis set to define prototypes. Defining multiple prototypes per target class will require significant modifications to this code.

## Prerequisites

Use Python 3. This project has been tested on a Linux CPU-only environment and with CUDA-capable GPUs.

This project has been tested with the following package settings:

* tensorflow==1.15.4
* scikit-learn==0.23.1
* Pillow==7.2.0
* matplotlib==3.2.2
* progress==1.5
* argparse==1.4.0
* mnist==0.2.2
* scipy==1.5.1
* pandas==1.0.5
* seaborn==0.10.1
* configparse==0.1.5

## Getting the datasets

Run

```
python3 get_data.py
```

To create and download all the datasets to this project directory. All datasets (stored as compressed Numpy arrays with extension .npz) will be stored inside `data/`. In total, 9 files will be created. The train, validation, and test splits will be saved. Versions of train, val, and test with Gaussian noise and random stroke noise separately will also be saved. Global variables inside `get_data.py` control this process. Noise samples have class label 0 while digits 0-9 have class labels 1-10.

## Training a model

Use `train_IUPG.py` to train a model. This script takes in a configuration file. A template, as well as two example config files, are provided in `configs/`. An example call is

```
python3 train_IUPG.py --config_files train_without_noise
```

The above call will train an IUPG model and save everything to the directory specified in `save_dir` in the config file. With this example config file, all models and results are saved to `cnn_results/no_noise/`. To train on a GPU on your machine, add the `--gpu_id [ID]` flag. For example, to train on GPU 0, do the following.

```
python3 train_IUPG.py --config_files train_without_noise --gpu_id 0
```

The resulting directory contains several self-explanatory CSV files that summarize performance. Additionally,

* `kmeans_plots/` will contain the prototype initializations which were discovered by clustering if that option was chosen.
* `models/` will contain the snapshot of the optimal model found during training.
* `perf_plots/` will contain accompanying plots of the summary files.
* `proto_plots/` will contain snapshot plots of the prototypes with a frequency that is specified in the config file.
* `tsne_plots/` will contain snapshot plots of some training data in the output vector space with a frequency that is specified in the config file. t-SNE is used to visualize this high dimensional space. You may need to edit the t-SNE parameters inside `IUPG_Builder.py` to get this to work well.

## Running inference

After the training process has concluded, use `inference.py` to produce predictions on new data. An example call is below.

```
python3 inference.py --model_dir cnn_results/no_noise --save_fp predictions/val_predictions.csv --npz_fp data/val.npz
```

This process will produce a self-explanatory `val_predictions.csv.csv` file with all of the results. In this case, an optimal threshold (to call noise samples) which maximizes accuracy will be calculated from the results. For proper test set analysis, use the optimal threshold that is calculated on the validation set (it will be printed to the terminal) and then apply it to the test set with the `--cust_thresh [THRESHOLD]` flag as in the following example.

```
python3 inference.py --model_dir cnn_results/no_noise --save_fp predictions/test_predictions.csv --npz_fp data/test.npz --cust_thresh [THRESHOLD]
```

You may also specify a GPU to use with the `--gpu_id [ID]` flag. If not, the process will run on CPU.

```
python3 inference.py --model_dir cnn_results/no_noise --save_fp predictions/val_predictions.csv --npz_fp data/val.npz --gpu_id 0
```

## Analyzing results

The script `analyze_scores.py` will enable you to analyze performance using the output of `inference.py`. You can compute the overall accuracy as well as the non-noise error (one minus the accuracy over all non-noise classes) at specified maximum false-positive rates. In this case, a false-positive is defined as a noise sample being classified as any non-noise class. Accordingly, the latter will be uninformative if your test set does not contain noise. An example call is shown below.

```
python3 analyze_scores.py --pred_fps predictions/val_predictions predictions/test_predictions --labels "Validation_Performance,Test_Performance" --ref_fprs 0.01,0.001,0.0001
```

In the above example, we are calculating the non-noise error on both the validation and test set at the 1%, 0.1%, and 0.01% FPR levels. This will display the accuracy at the threshold which was pre-applied from `inference.py`. To compute the accuracy at a custom threshold, use the `--cust_thresh` flag like so.

```
python3 analyze_scores.py --pred_fps predictions/val_predictions predictions/test_predictions --labels "Validation_Performance,Test_Performance" --ref_fprs 0.01,0.001,0.0001 --cust_thresh [THRESHOLD]
```

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details
