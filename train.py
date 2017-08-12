from __future__ import print_function
import os
import sys
import scipy
import scipy
import logging
import scipy.io
import threading
import subprocess
import numpy as np
import pandas as pd
from VGG import VGG
import seaborn as sns
from skimage import io
from io import BytesIO
import tensorflow as tf
from scipy import ndimage
from six import string_types
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multithrdloader import MultiThrdLoader

TRAIN_BATCHES=50
TEST_BATCHES=50

def main():

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    with tf.Session() as sess:

        logging.info("create VGG")
        vgg = VGG("models/imagenet-vgg-verydeep-19.mat")
        logging.info("loading VGG")
        graph, layers = vgg.create_graph(tf.placeholder(tf.float32, shape=[None, 256, 256, 3]))

        # add two fc layers
        net = graph["pool5"]
        shape = net.shape.as_list()

        net = tf.layers.dense(tf.reshape(net, [-1, (shape[1] * shape[2] * shape[3])]), 2048)
        net = tf.layers.dropout(net)
        net = tf.layers.dense(net, 2048)
        net = tf.layers.dropout(net)
        net = tf.layers.dense(net, 17)
        out = tf.sigmoid(net)

        y = tf.placeholder(tf.float32, shape=[None, 17])
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=out)
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

        output = tf.round(out)
        correctness = tf.equal(output, y)
        accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))

        sess.run(tf.global_variables_initializer())
        logging.info("initialized")

        logging.info("create loader")
        train_loader = MultiThrdLoader(sess, "data", "train-jpg", "train_v2.csv", list(range(30000)), num_thrds=100)
        train_batch64 = train_loader.get_batch_op(64)

        test_loader = MultiThrdLoader(sess, "data", "train-jpg", "train_v2.csv", list(range(30000, 40000)), num_thrds=50)
        test_batch64 = test_loader.get_batch_op(64)

        # training
        for i in range(TRAIN_BATCHES):
            batch_X, batch_y, _ = sess.run(train_batch64)
            acc, predict = sess.run([accuracy, output], feed_dict={graph["input"]: batch_X, y: batch_y})
            sess.run(train_step, feed_dict={graph["input"]: batch_X, y: batch_y})
            logging.info("batch Accuracy: %f" % acc)
        # testing
        test_accuracy = 0
        for i in range(TEST_BATCHES):
            batch_X, batch_y, _ = sess.run(test_batch64)
            acc, predict = sess.run([accuracy, output], feed_dict={graph["input"]: batch_X, y: batch_y})
            logging.info("test batch Accuracy: %f" % acc)
            test_accuracy += acc
        logging.info("test average accuracy: %f" % (test_accuracy / TEST_BATCHES))

        logging.info("finished")

        test_loader.stop()
        train_loader.stop()

main()
