import tensorflow as tf
import sys
import os
import scipy
import _thread
import threading
import pandas as pd
import numpy as np
from skimage import io
from scipy import ndimage
import threading
import scipy
import scipy.io
from io import BytesIO

class MultiThrdLoader(object):

    def __init__(self, sess, data_root, img_dir, label_csv, file_idxs, \
                    data_shape=(256, 256, 3), label_shape=(17), num_thrds=10):
        """
        Multiple background worker threads loading the images
        sess: a Tensorflow session
        dataroot: the root path of the data folder
        img_dir: the relative path of the directory contains images w.r.t dataroot
        label_csv: the relative path of the csv file for labels w.r.t dataroot
        data_shape: shape of each data item
        num_thrds: number of worker threads loading images
        """
        self.data_root = data_root
        self.label_csv = os.path.join(data_root, label_csv)
        self.img_dir = os.path.join(data_root, img_dir)
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.sess = sess

        self.coord = tf.train.Coordinator()

        # Create a randomly shffled file name queue
        self.y, self.file_ids = self._process_labels()
        self.file_idx_q = tf.RandomShuffleQueue(10000, 0, dtypes=[tf.int32], shapes=[()])
        self.eq_fn_op = self.file_idx_q.enqueue_many([file_idxs])
        self.fn_qr = tf.train.QueueRunner(self.file_idx_q, [self.eq_fn_op])
        fn_thrds = self.fn_qr.create_threads(sess, coord=self.coord, start=True)

        # Create the data batch FIFO queue
        self.data_q = tf.FIFOQueue(500, dtypes=[tf.float32, tf.float32, tf.int32], shapes=[data_shape, label_shape, ()])
        self.X_holder = tf.placeholder(tf.float32)
        self.y_holder = tf.placeholder(tf.float32)
        self.idx_holder = tf.placeholder(tf.int32)
        self.enqueue = self.data_q.enqueue([self.X_holder, self.y_holder, self.idx_holder])
        self.fdx_dequeue = self.file_idx_q.dequeue()
        self.threads = [threading.Thread(target=self._load_worker, args=(sess,)) for i in range(num_thrds)]
        for t in self.threads:
            t.start()
        self.threads += fn_thrds

    def _process_labels(self):
        labels_df = pd.read_csv(self.label_csv)

        # Find all availabe tags and map to index
        label_list = []
        for tag_str in labels_df.tags.values:
            labels = tag_str.split(" ")
            for label in labels:
                if label not in label_list:
                    label_list.append(label)
        label_map = {}
        for i in range(len(label_list)):
            label_map[label_list[i]] = i
        print("All available labels: ", label_map)

        # create one hot vectors
        y = np.zeros((labels_df.shape[0], len(label_list)))
        file_ids = []
        for i in range(labels_df.shape[0]):
            labels = labels_df.tags.values[i].split(" ")
            file_ids.append(labels_df.image_name.values[i])
            for label in labels:
                y[i, label_map[label]] = 1
        return y, file_ids

    def _load_worker(self, sess):
        while not self.coord.should_stop():
            # dequeue one filename from the file name queue
            idx = sess.run(self.fdx_dequeue)
            # load the image
            X = np.ndarray(self.data_shape)
            y = np.ndarray(self.label_shape)
            file_path = os.path.join(self.img_dir, self.file_ids[idx] + ".jpg")
            X = io.imread(file_path)
            y = self.y[idx]
            try:
                sess.run(self.enqueue, feed_dict={
                    self.X_holder: X,
                    self.y_holder: y,
                    self.idx_holder: idx
                    })
            except tf.errors.CancelledError:
                return

    def get_batch_op(self, n):
        return self.data_q.dequeue_many(n)

    def stop(self):
        self.coord.request_stop()
        d_q_clos = self.data_q.close(cancel_pending_enqueues=True)
        self.sess.run(d_q_clos)
        self.coord.join(self.threads)

