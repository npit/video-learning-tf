import tensorflow as tf
from numpy import *
from  utils_ import *
# Alexnet and weights from
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
# and
# https://github.com/ethereon/caffe-tensorflow

class dcnn(Trainable):
    input = None
    output = None

    # helper convolution definer
    def conv(self,input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1,name=None):
        # print (kernel.shape)
        with tf.name_scope(name):
            c_i = input.get_shape()[-1]
            assert c_i%group==0
            assert c_o%group==0
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


            if group==1:
                conv = convolve(input, kernel)
            else:
                input_groups =  tf.split(input, group, 3,name="conv_internal")   #tf.split(3, group, input)
                kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
        return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:], name=name+"_final_reshape")

    def get_io(self):
        return self.input, self.output
    def get_input(self):
        return self.input
    def get_output(self):
        return self.output

        # specify the layers
    def create(self, xdim, weightsFile, num_classes, final_layer ="prob"):
        net_data = load(open(weightsFile, "rb"), encoding="latin1").item()
        #net_data = load("bvlc_alexnet.npy").item()
        self.input = tf.placeholder(tf.float32, (None,) + xdim, name='input_frames')

        with tf.name_scope("dcnn") as scope:
            # conv1
            # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
            k_h = 11;
            k_w = 11;
            c_o = 96;
            s_h = 4;
            s_w = 4
            conv1W = tf.Variable(net_data["conv1"][0],name="conv1W")
            conv1b = tf.Variable(net_data["conv1"][1],name="conv1b")
            conv1_in = self.conv(self.input, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1,name="conv1")
            conv1 = tf.nn.relu(conv1_in,name="relu1")

            # lrn1
            # lrn(2, 2e-05, 0.75, name='norm1')
            radius = 2;
            alpha = 2e-05;
            beta = 0.75;
            bias = 1.0
            lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias,name="lrn1")

            # maxpool1
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
            k_h = 3;
            k_w = 3;
            s_h = 2;
            s_w = 2;
            padding = 'VALID'
            maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding,name="pool1")

            # conv2
            # self.conv(5, 5, 256, 1, 1, group=2, name='conv2')
            k_h = 5;
            k_w = 5;
            c_o = 256;
            s_h = 1;
            s_w = 1;
            group = 2
            conv2W = tf.Variable(net_data["conv2"][0],name="conv2W")
            conv2b = tf.Variable(net_data["conv2"][1],name="conv2b")
            conv2_in = self.conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group,name="conv2")
            conv2 = tf.nn.relu(conv2_in,name="relu_2")

            # lrn2
            # lrn(2, 2e-05, 0.75, name='norm2')
            radius = 2;
            alpha = 2e-05;
            beta = 0.75;
            bias = 1.0
            lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias,name="lrn2")

            # maxpool2
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
            k_h = 3;
            k_w = 3;
            s_h = 2;
            s_w = 2;
            padding = 'VALID'
            maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding,name="pool2")

            # conv3
            # self.conv(3, 3, 384, 1, 1, name='conv3')
            k_h = 3;
            k_w = 3;
            c_o = 384;
            s_h = 1;
            s_w = 1;
            group = 1
            conv3W = tf.Variable(net_data["conv3"][0],name="conv3W")
            conv3b = tf.Variable(net_data["conv3"][1],name="conv3b")
            conv3_in = self.conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group,name="conv3")
            conv3 = tf.nn.relu(conv3_in,name="relu3")

            # conv4
            # self.conv(3, 3, 384, 1, 1, group=2, name='conv4')
            k_h = 3;
            k_w = 3;
            c_o = 384;
            s_h = 1;
            s_w = 1;
            group = 2
            conv4W = tf.Variable(net_data["conv4"][0],name="conv4W")
            conv4b = tf.Variable(net_data["conv4"][1],name="conv3b")
            conv4_in = self.conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group,name="conv4")
            conv4 = tf.nn.relu(conv4_in,name="relu4")

            # conv5
            # conv(3, 3, 256, 1, 1, group=2, name='conv5')
            k_h = 3;
            k_w = 3;
            c_o = 256;
            s_h = 1;
            s_w = 1;
            group = 2
            conv5W = tf.Variable(net_data["conv5"][0],name="conv5W")
            conv5b = tf.Variable(net_data["conv5"][1],name="conv5b")
            conv5_in = self.conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group,name="conv5")
            conv5 = tf.nn.relu(conv5_in,name="relu5")

            # maxpool5
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
            k_h = 3;
            k_w = 3;
            s_h = 2;
            s_w = 2;
            padding = 'VALID'
            maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding,name="pool5")

            # add all variables so far to regular training speed
            self.train_regular.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope))

            # fc6
            # fc(4096, name='fc6')
            fc6W = tf.Variable(net_data["fc6"][0],name="fc6W")
            fc6b = tf.Variable(net_data["fc6"][1],name="fc6b")
            fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))],name="fc6_relu_reshape"), fc6W, fc6b,name="fc6")

            # add fc6
            self.train_regular.extend([fc6W, fc6b])

            if final_layer == "fc6":
                self.output = fc6
                return
            # fc7
            # fc(4096, name='fc7')
            fc7W = tf.Variable(net_data["fc7"][0],name="fc7W")
            fc7b = tf.Variable(net_data["fc7"][1],name="fc7b")
            fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b,name="fc7")

            # add fc6
            self.train_regular.extend([fc7W, fc7b])

            if final_layer == "fc7":
                self.output = fc7
                return
            # original imagenet fc8 layer
            # ---------------------------
            # fc8
            # fc(1000, relu=False, name='fc8')
            # fc8W = tf.Variable(net_data["fc8"][0],name="fc8W")
            # fc8b = tf.Variable(net_data["fc8"][1],name="fc8b")
            # fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b,name="fc8")
            # ---------------------------

            # new fc8 layer for custom num of classes
            # ---------------------------
            # fc8
            # stdev of 0.05 seems to work s.t. loss does not diverge to hell

            w_init = tf.truncated_normal((4096 , num_classes), stddev=0.05, name="fc8_init")
            b_init = tf.constant(0.1, shape =(num_classes,), name = "fc8_bias")

            fc8W = tf.Variable(w_init, name="fc8W")
            fc8b = tf.Variable(b_init, name="fc8b")
            self.output = tf.nn.xw_plus_b(fc7, fc8W, fc8b,name="fc8")



            # fc8 is new layer, train in a modified manner
            self.train_modified.extend([fc8W, fc8b])

            # ---------------------------

            # omitting the rest. Softmax will be applied during  the loss calculation
            # or the evaluation procedure
            # if final_layer == "fc8":
            #     return x, fc6

            # prob
            # softmax(name='prob'))
            # prob = tf.nn.softmax(fc8,name="softmax")

