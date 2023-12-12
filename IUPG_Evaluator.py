"""
IUPG_Evaluator.py

Load and use a saved IUPG model.

@author: Brody Kutt (bkutt@paloaltonetworks.com)
Copyright (c) 2020 Palo Alto Networks
"""

import os
import numpy as np
import tensorflow as tf
from progress.bar import Bar
from IUPG_Input import IUPG_Input

# Set this to true to debug GPU usage
LOG_DEVICE_PLACEMENT = False


class IUPG_Evaluator(object):
    """
    Class to load and use a saved IUPG instance.
    """
    def __init__(
        self,
        # The identifiable path to the model directory
        iupg_path,
    ):

        self.iupg_path = iupg_path
        self.load_model()

    def load_model(self):
        """
        Load the IUPG graph.
        """
        model_dir = os.path.join(self.iupg_path, "models",
                                 os.path.basename(self.iupg_path))
        meta_path = os.path.join(self.iupg_path, "models",
                                 os.path.basename(self.iupg_path) + ".meta")
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.compat.v1.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=LOG_DEVICE_PLACEMENT,
            )
            self.sess = tf.compat.v1.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.compat.v1.train.import_meta_graph(meta_path)
                saver.restore(self.sess, model_dir)
                # Get the input placeholders from the graph by name
                self.X_batch = graph.get_operation_by_name(
                    "X_batch").outputs[0]
                self.y_batch = graph.get_operation_by_name(
                    "y_batch").outputs[0]
                self.drop_keep_prob = graph.get_operation_by_name(
                    "drop_keep_prob").outputs[0]
                self.run_samps = graph.get_operation_by_name(
                    "run_samps").outputs[0]
                self.run_protos = graph.get_operation_by_name(
                    "run_protos").outputs[0]
                self.given_U_protos = graph.get_operation_by_name(
                    "given_U_protos").outputs[0]
                # Tensors we want to evaluate
                self.protos = graph.get_operation_by_name(
                    "proto_init/protos").outputs[0]
                self.proto_U_vec = graph.get_operation_by_name(
                    "network/proto_U_vec").outputs[0]
                self.sig_D = graph.get_operation_by_name(
                    "output/sig_D").outputs[0]

    def proto_feed_dict(self):
        """
        Define the feed dict that should be used when evaluating prototypes.
        """
        return {
            self.run_protos: True,
            self.run_samps: False,
            self.drop_keep_prob: 1.0,
        }

    def get_U_protos(self):
        """
        Return the value of the prototypes projected onto the U vector space.
        """
        return self.sess.run(self.proto_U_vec, self.proto_feed_dict())

    def get_protos(self):
        """
        Return the value of the prototypes at lowest level.
        """
        return np.squeeze(self.sess.run(self.protos, self.proto_feed_dict()))

    def create_feed_dict(
        self,
        batch,
        run_samps=True,
        run_protos=True,
        include_y=False,
        U_protos=None,
    ):
        """
        Define a dictionary to feed into the TF session.
        """
        feed_dict = {}
        feed_dict[self.X_batch] = batch["X"]
        feed_dict[self.drop_keep_prob] = 1.0
        feed_dict[self.run_samps] = run_samps
        feed_dict[self.run_protos] = run_protos
        if include_y:
            feed_dict[self.y_batch] = batch["y"]
        if U_protos is not None:
            feed_dict[self.given_U_protos] = U_protos
        return feed_dict

    def predict(self, X=[], U_protos=None, batch_size=32, v=False):
        """
        Produce predictions for the data in X.
        """
        n_samps = len(X)
        # Generate batches for the data
        X_iter = IUPG_Input(X).get_X_batch_iter(batch_size=batch_size)
        if U_protos is None:
            U_protos = self.get_U_protos()

        # Collect the predictions here
        all_sig_D = []
        # Start predicting batches
        prev_size = 0
        if v:
            bar = Bar("Progress... ", max=n_samps)
        for batch in X_iter:
            batch_sig_D = self.sess.run(
                self.sig_D,
                self.create_feed_dict(batch,
                                      run_protos=False,
                                      U_protos=U_protos),
            )
            all_sig_D.extend(batch_sig_D.tolist())
            if v:
                bar.next(len(all_sig_D) - prev_size)
                prev_size = len(all_sig_D)
        if v:
            bar.finish()
        return np.array(all_sig_D)
