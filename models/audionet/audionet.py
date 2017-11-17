import tensorflow as tf
from numpy import *
from  utils_ import *
from tf_util import *
# create a network to train audio spectrograms
# model based on tf's cifar10 model: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
class audionet(Trainable):
    input = None
    output = None

    def create(self, input_shape, num_classes):
        """
        Define a simple convolutional net for small spectrogram images
        :param dataset:
        :param settings:
        :return:
        """
        self.input = tf.placeholder(tf.float32, [None] + list(input_shape))

        with tf.variable_scope("conv1") as scope:
            # conv params
            num_kernels = 64
            height, width, depth = 5, 5, 3
            strides = [1, 1, 1, 1]
            padding = "SAME"
            conv1 = make_conv(self.input, [height, width, depth, num_kernels], strides, scope.name, padding=padding)

        with tf.variable_scope("pool1") as scope:
            # pool params
            window = [1, 3, 3, 1 ]
            strides = [1, 2, 2, 1]
            padding="SAME"
            pool1 = make_pool(conv1, window, strides, padding, scope.name)

        with tf.variable_scope("conv2") as scope:
            num_kernels = 64
            height, width, depth = 5, 5, 64
            strides = [1, 1, 1, 1]
            padding = "SAME"
            conv2= make_conv(pool1, [height, width, depth, num_kernels], strides, scope.name, padding=padding)

        with tf.variable_scope("pool2") as scope:
            window = [1, 3, 3, 1 ]
            strides = [1, 2, 2, 1]
            padding="SAME"
            pool2 = make_pool(conv2, window, strides, padding, scope.name)

        with tf.variable_scope("conv3") as scope:
            num_kernels = 64
            height, width, depth = 11, 11, 64
            strides = [1, 2, 2, 1]
            padding = "SAME"
            conv3 = make_conv(pool2, [height, width, depth, num_kernels], strides, scope.name, padding=padding)

        with tf.variable_scope("pool3") as scope:
            window = [1, 3, 3, 1 ]
            strides = [1, 2, 2, 1]
            padding="SAME"
            pool4 = make_pool(conv3, window, strides, padding, scope.name)

        depth_dim = 1
        for x in pool4.shape[1:]:
            depth_dim *= int(x)
        # pre_fc_vector = vectorize(pool4, batch_size, depth_dim)
        pre_fc_vector = vectorize(pool4, depth_dim)

        with tf.variable_scope("fc1") as scope:
            output_dim = num_classes
            self.output = convert_dim_fc(pre_fc_vector, output_dim, scope.name)

        # all variables to the fast track
        self.ignorable_variable_names.extend([])


    def get_io(self):
        return self.input, self.output





