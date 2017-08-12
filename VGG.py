import scipy
import tensorflow as tf
from skimage import io
from scipy import ndimage
import threading
import scipy
import scipy.io
from io import BytesIO
import numpy as np

class VGG(object):

    def __init__(self, path, sess=None):
        self.content_layer = None
        self.style_layer = None
        self.path = path
        self.weights = scipy.io.loadmat(self.path)['layers']
        self.sess = sess

    def create_graph(self, input_layer, include_fc=False):
        """
        The pretrained model contains the layer name and layer type (i.e. pool, conv etc.)
        vgg_layers[0][30][0][0][0][0] # to access layer name
        vgg_layers[0][30][0][0][1][0] # to access layer type

        Note that the fully connected layers and the softmax are not required for this task,
        therefore we will skip it.
        The fully connected layers have name fc* (It's type is conv though).
        """

        vgg_layers = self.weights
        num_layers = len(vgg_layers[0])

        graph = {}
        layer_names = []
        graph["input"] = input_layer
        prev = "input"
        layer_names.append("input")

        for idx in range(num_layers):

            layer_name = vgg_layers[0][idx][0][0][0][0]
            layer_type = vgg_layers[0][idx][0][0][1][0]

            if layer_name[:2] == "fc":
                break                     # stop before adding the first fc layer

            layer_names.append(layer_name)

            if layer_type == "conv":
                W_val = vgg_layers[0][idx][0][0][2][0][0]
                b_val = vgg_layers[0][idx][0][0][2][0][1]

                W = tf.Variable(tf.constant(W_val), dtype=tf.float32, trainable=False)
                b = tf.Variable(tf.constant(np.reshape(b_val, (b_val.size))), dtype=tf.float32, trainable=False)

                if self.sess != None:
                    self.sess.run(W.assign(W_val))
                    self.sess.run(b.assign(np.reshape(b_val, (b_val.size))))

                graph[layer_name] = tf.nn.conv2d(graph[prev], filter=W,
                                            strides=[1, 1, 1, 1], padding="SAME") + b
            elif layer_type == "relu":
                graph[layer_name] = tf.nn.relu(graph[prev])
            elif layer_type == "pool":
                # according to the paper, average pooling behaves better
                graph[layer_name] = tf.nn.avg_pool(graph[prev], ksize=[1, 2, 2, 1],
                                                   strides=[1, 2, 2, 1], padding="SAME")
            else:
                raise Exception("Unknown layer")

            prev = layer_name
        return graph, layer_names

    def get_layer_names(self):
        return self.layer_names
