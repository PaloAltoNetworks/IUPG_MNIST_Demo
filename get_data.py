"""
get_data.py

Create and save all MNIST-based datasets to disk.

@author: Brody Kutt (bkutt@paloaltonetworks.com)
Copyright (c) 2020 Palo Alto Networks
"""

import os
import utils
import mnist
import numpy as np
from progress.bar import Bar
from PIL import Image, ImageDraw
from random import randint, uniform, seed
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.ndimage import median_filter, gaussian_filter, binary_dilation

seed(utils.random_seed())
np.random.seed(utils.random_seed())
if not os.path.isdir("data"):
    os.mkdir("data")

# Size of validation and test sets
VAL_SIZE = 10000
TEST_SIZE = 10000

# Number of training, validation, test noise images to add
N_TRAIN_NOISE_IMAGES = 50000
N_VAL_NOISE_IMAGES = 10000
N_TEST_NOISE_IMAGES = 10000

# Minimum confidence of the RF used during stroke noise generation
RF_CONF = 0.3


def create_gauss_noise_images(n_train,
                              n_val,
                              n_test,
                              scales=np.arange(0.7, 2.2, 0.1)):
    """
    Construct Gaussian noise samples.
    """
    print("--> Constructing training Gaussian noise images...")
    train_rnd_sample = None
    for scale in scales:
        new_rnd_sample = np.random.normal(
            loc=-1,
            scale=scale,
            size=(int(n_train / float(len(scales))), 22, 22),
        )
        new_rnd_sample = np.pad(
            new_rnd_sample,
            pad_width=[[0, 0], [3, 3], [3, 3]],
            mode="constant",
            constant_values=0.0,
        )
        if train_rnd_sample is None:
            train_rnd_sample = new_rnd_sample
        else:
            train_rnd_sample = np.vstack((train_rnd_sample, new_rnd_sample))
    print(("----> Shape of training Gaussian noise samples: %s" %
           str(train_rnd_sample.shape)))

    print("--> Constructing validation Gaussian noise images...")
    val_rnd_sample = None
    for scale in scales:
        new_rnd_sample = np.random.normal(
            loc=-1,
            scale=scale,
            size=(int(n_val / float(len(scales))), 22, 22))
        new_rnd_sample = np.pad(
            new_rnd_sample,
            pad_width=[[0, 0], [3, 3], [3, 3]],
            mode="constant",
            constant_values=0.0,
        )
        if val_rnd_sample is None:
            val_rnd_sample = new_rnd_sample
        else:
            val_rnd_sample = np.vstack((val_rnd_sample, new_rnd_sample))
    print(("----> Shape of validation Gaussian noise samples: %s" %
           str(val_rnd_sample.shape)))

    print("--> Constructing test Gaussian noise images...")
    test_rnd_sample = None
    for scale in scales:
        new_rnd_sample = np.random.normal(
            loc=-1,
            scale=scale,
            size=(int(n_test / float(len(scales))), 22, 22))
        new_rnd_sample = np.pad(
            new_rnd_sample,
            pad_width=[[0, 0], [3, 3], [3, 3]],
            mode="constant",
            constant_values=0.0,
        )
        if test_rnd_sample is None:
            test_rnd_sample = new_rnd_sample
        else:
            test_rnd_sample = np.vstack((test_rnd_sample, new_rnd_sample))
    print(("----> Shape of test Gaussian noise samples: %s" %
           str(test_rnd_sample.shape)))

    print("--> Running median filters...")
    for i in range(len(train_rnd_sample)):
        train_rnd_sample[i, :, :] = median_filter(train_rnd_sample[i, :, :],
                                                  size=(2, 2))
    for i in range(len(val_rnd_sample)):
        val_rnd_sample[i, :, :] = median_filter(val_rnd_sample[i, :, :],
                                                size=(2, 2))
    for i in range(len(test_rnd_sample)):
        test_rnd_sample[i, :, :] = median_filter(test_rnd_sample[i, :, :],
                                                 size=(2, 2))

    print("--> Clipping...")
    train_noise = np.clip(train_rnd_sample, a_min=0.0, a_max=1.0)
    val_noise = np.clip(val_rnd_sample, a_min=0.0, a_max=1.0)
    test_noise = np.clip(test_rnd_sample, a_min=0.0, a_max=1.0)

    print("--> Subtracting per-image means...")
    for i in range(len(train_noise)):
        train_noise[i, :, :] -= np.mean(train_noise[i, :, :])
    for i in range(len(val_noise)):
        val_noise[i, :, :] -= np.mean(val_noise[i, :, :])
    for i in range(len(test_noise)):
        test_noise[i, :, :] -= np.mean(test_noise[i, :, :])
    return train_noise, val_noise, test_noise


def create_stroke_noise_images(n_train, n_val, n_test, rf):
    """
    Construct random stroke images.
    """
    # Possible strokes to choose from
    possible_strokes = [
        "line",
        "chord",
        "ellipse",
        "polygon",
        "rectangle",
        "arc",
        "pieslice",
    ]

    def draw_random_stroke(d, stroke_type_override=None):
        """
        Draw a single stroke onto d.
        """
        stroke_details = {}
        if stroke_type_override is None:
            stroke_type = possible_strokes[randint(0, 6)]
        else:
            stroke_type = stroke_type_override
        stroke_details["type"] = stroke_type
        bb_width = randint(5, 22)
        bb_height = randint(5, 22)
        bb_x0 = randint(0, 22 - bb_width)
        bb_y0 = randint(0, 22 - bb_height)
        bb = [bb_x0, bb_y0, bb_x0 + bb_width, bb_y0 + bb_height]
        stroke_details["bb"] = bb
        if stroke_type == "line":
            width = randint(1, 2)
            stroke_details["width"] = width
            d.line(bb, fill=1, width=width)
        elif stroke_type == "chord":
            start, end = randint(0, 360), randint(0, 360)
            stroke_details["start"], stroke_details["end"] = start, end
            d.chord(bb, start=start, end=end, fill=0, outline=1)
        elif stroke_type == "ellipse":
            d.ellipse(bb, fill=0, outline=1)
        elif stroke_type == "polygon":
            n_verts = randint(3, 6)
            points = [(randint(0, 22), randint(0, 22)) for i in range(n_verts)]
            stroke_details["n_vertices"] = n_verts
            stroke_details["points"] = points
            d.polygon(points, fill=0, outline=1)
        elif stroke_type == "rectangle":
            d.rectangle(bb, fill=0, outline=1)
        elif stroke_type == "arc":
            start, end = randint(0, 360), randint(0, 360)
            stroke_details["start"], stroke_details["end"] = start, end
            d.arc(bb, start=start, end=end, fill=1)
        elif stroke_type == "pieslice":
            start, end = randint(0, 360), randint(0, 360)
            stroke_details["start"], stroke_details["end"] = start, end
            d.pieslice(bb, start=start, end=end, fill=0, outline=1)
        else:
            assert stroke_type in possible_strokes
        return stroke_details

    def generate_stroke_image():
        """
        Generate a single new random stroke image.
        """
        no_valid_image_yet = True
        while no_valid_image_yet:
            # Draw random strokes
            n_strokes = randint(1, 5)
            rnd_image = Image.new(mode="1", size=(22, 22), color=0)
            d = ImageDraw.Draw(rnd_image)
            for j in range(n_strokes):
                if n_strokes == 1:
                    st_details = draw_random_stroke(
                        d, stroke_type_override="polygon")
                else:
                    st_details = draw_random_stroke(d)
            assert not (n_strokes == 1 and st_details["type"] != "polygon")
            # Pad to 28x28 size
            rnd_image = np.pad(
                rnd_image,
                pad_width=[[3, 3], [3, 3]],
                mode="constant",
                constant_values=0.0,
            )
            dilation_flag = randint(0, 2)
            if dilation_flag == 1:
                rnd_image = binary_dilation(rnd_image, iterations=1)
            elif dilation_flag == 2:
                rnd_image = binary_dilation(rnd_image, iterations=2)
            # Convert to float
            rnd_image = rnd_image.astype("float")
            # Run Gaussian filter
            sigma = uniform(0.35, 0.75)
            rnd_image = gaussian_filter(rnd_image, sigma=sigma)
            # Clip between 0 and 1
            rnd_image = np.clip(rnd_image, a_min=0.0, a_max=1.0)
            # Subtract away the mean
            rnd_image -= np.mean(rnd_image)
            # Compute prediction of RF and decide if to keep
            y_pred = rf.predict_proba(rnd_image.reshape((1, 28 * 28)))
            if max(y_pred[0]) >= RF_CONF:
                no_valid_image_yet = True
            else:
                no_valid_image_yet = False
        return rnd_image

    train_noise = []
    print("--> Constructing training stroke noise images...")
    bar = Bar("Progress... ", max=n_train)
    for i in range(n_train):
        train_noise.append(generate_stroke_image())
        bar.next()
    bar.finish()
    train_noise = np.array(train_noise)
    print(("--> Shape of training stroke noise samples: %s" %
           str(train_noise.shape)))

    print("Constructing validation stroke noise images...")
    val_noise = []
    bar = Bar("Progress... ", max=n_val)
    for i in range(n_val):
        val_noise.append(generate_stroke_image())
        bar.next()
    bar.finish()
    val_noise = np.array(val_noise)
    print(("--> Shape of validation stroke noise samples: %s" %
           str(val_noise.shape)))

    print("Constructing test stroke noise images...")
    test_noise = []
    bar = Bar("Progress... ", max=n_test)
    for i in range(n_test):
        test_noise.append(generate_stroke_image())
        bar.next()
    bar.finish()
    test_noise = np.array(test_noise)
    print(
        ("--> Shape of test stroke noise samples: %s" % str(test_noise.shape)))
    return train_noise, val_noise, test_noise


if __name__ == "__main__":
    print(("-" * 80))
    print("Loading training and testing data...")
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    print("Combining train and test data...")
    all_images = np.concatenate((train_images, test_images), axis=0)
    all_labels = np.concatenate((train_labels, test_labels), axis=0)

    print("Bounding images between 0 and 1..")
    preprocessed_all_images = all_images / 255.0

    print("Subtracting per-image means...")
    for i in range(len(preprocessed_all_images)):
        preprocessed_all_images[i] -= np.mean(preprocessed_all_images[i])

    print("Randomly sampling TTV...")
    X_leftover, X_test, y_leftover, y_test = train_test_split(
        preprocessed_all_images,
        all_labels,
        test_size=TEST_SIZE,
        random_state=utils.random_seed(),
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_leftover,
        y_leftover,
        test_size=VAL_SIZE,
        random_state=utils.random_seed(),
    )

    # Add one to labels to account for the noise class
    y_train += 1
    y_val += 1
    y_test += 1

    print("\nTrain data:")
    for i in range(10):
        print(
            ("--> Number of %d samples: %d" % (i, len(y_train[y_train == i]))))
    print(("--> Number of samples total: %d" % len(y_train)))

    print("\nVal data:")
    for i in range(10):
        print(("--> Number of %d samples: %d" % (i, len(y_val[y_val == i]))))
    print(("--> Number of samples total: %d" % len(y_val)))

    print("\nTest data:")
    for i in range(10):
        print(("--> Number of %d samples: %d" % (i, len(y_test[y_test == i]))))
    print(("--> Number of samples total: %d" % len(y_test)))

    print("Saving new NPZs to disk...")
    int2label = {
        0: "Noise",
        1: "0",
        2: "1",
        3: "2",
        4: "3",
        5: "4",
        6: "5",
        7: "6",
        8: "7",
        9: "8",
        10: "9",
    }
    np.savez_compressed(
        os.path.join("data", "train.npz"),
        X=X_train,
        y=y_train,
        int2label=int2label,
    )
    print(("--> Saved train.npz"))
    np.savez_compressed(os.path.join("data", "val.npz"),
                        X=X_val,
                        y=y_val,
                        int2label=int2label)
    print(("--> Saved val.npz"))
    np.savez_compressed(
        os.path.join("data", "test.npz"),
        X=X_test,
        y=y_test,
        int2label=int2label,
    )
    print(("--> Saved test.npz"))

    ###########################################################################

    print("\nCreating dataset with Gaussian noise...")
    train_noise, val_noise, test_noise = create_gauss_noise_images(
        N_TRAIN_NOISE_IMAGES,
        N_VAL_NOISE_IMAGES,
        N_TEST_NOISE_IMAGES,
    )
    print("Combining datasets...")
    new_train_images = np.vstack((X_train, train_noise))
    new_train_labels = np.concatenate((y_train, ([0] * len(train_noise))))
    print(
        ("--> New size of training images: %s" % str(new_train_images.shape)))
    assert len(new_train_images) == len(new_train_labels)

    new_val_images = np.vstack((X_val, val_noise))
    new_val_labels = np.concatenate((y_val, ([0] * len(val_noise))))
    print(
        ("--> New size of validation images: %s" % str(new_val_images.shape)))
    assert len(new_val_images) == len(new_val_labels)

    new_test_images = np.vstack((X_test, test_noise))
    new_test_labels = np.concatenate((y_test, ([0] * len(test_noise))))
    print(("--> New size of testing images: %s" % str(new_test_images.shape)))
    assert len(new_test_images) == len(new_test_labels)

    print("Saving new NPZs to disk...")
    np.savez_compressed(
        os.path.join("data", "train-gauss.npz"),
        X=new_train_images,
        y=new_train_labels,
        int2label=int2label,
    )
    print(("--> Saved train_gauss.npz"))
    np.savez_compressed(
        os.path.join("data", "val-gauss.npz"),
        X=new_val_images,
        y=new_val_labels,
        int2label=int2label,
    )
    print(("--> Saved val_gauss.npz"))
    np.savez_compressed(
        os.path.join("data", "test-gauss.npz"),
        X=new_test_images,
        y=new_test_labels,
        int2label=int2label,
    )
    print(("--> Saved test_gauss.npz"))

    ###########################################################################

    print("\nTraining RF to be used during stroke noise generation...")
    rf = RandomForestClassifier(n_estimators=200, criterion="entropy")
    rf.fit(X_train.reshape(len(X_train), 28 * 28), y_train)

    print("Computing test set predictions...")
    rf_scores = rf.predict_proba(X_test.reshape(len(X_test), 28 * 28))
    predicted = []
    for score_vec in rf_scores:
        if max(score_vec) >= RF_CONF:
            predicted.append(np.argmax(score_vec) + 1)
        else:
            predicted.append(0)
    print(("--> Accuracy: %f" % accuracy_score(y_test, predicted)))

    print("\nCreating dataset with stroke noise...")
    train_noise, val_noise, test_noise = create_stroke_noise_images(
        N_TRAIN_NOISE_IMAGES, N_VAL_NOISE_IMAGES, N_TEST_NOISE_IMAGES, rf)
    print("Combining datasets...")
    new_train_images = np.vstack((X_train, train_noise))
    new_train_labels = np.concatenate((y_train, ([0] * len(train_noise))))
    print(
        ("--> New size of training images: %s" % str(new_train_images.shape)))
    assert len(new_train_images) == len(new_train_labels)

    new_val_images = np.vstack((X_val, val_noise))
    new_val_labels = np.concatenate((y_val, ([0] * len(val_noise))))
    print(
        ("--> New size of validation images: %s" % str(new_val_images.shape)))
    assert len(new_val_images) == len(new_val_labels)

    new_test_images = np.vstack((X_test, test_noise))
    new_test_labels = np.concatenate((y_test, ([0] * len(test_noise))))
    print(("--> New size of testing images: %s" % str(new_test_images.shape)))
    assert len(new_test_images) == len(new_test_labels)

    print("Saving new NPZs to disk...")
    np.savez_compressed(
        os.path.join("data", "train-stroke.npz"),
        X=new_train_images,
        y=new_train_labels,
        int2label=int2label,
    )
    print(("--> Saved train-stroke.npz"))
    np.savez_compressed(
        os.path.join("data", "val-stroke.npz"),
        X=new_val_images,
        y=new_val_labels,
        int2label=int2label,
    )
    print(("--> Saved val-stroke.npz"))
    np.savez_compressed(
        os.path.join("data", "test-stroke.npz"),
        X=new_test_images,
        y=new_test_labels,
        int2label=int2label,
    )
    print(("--> Saved test-stroke.npz"))
    print("\nExiting...")
    print(("-" * 80))
