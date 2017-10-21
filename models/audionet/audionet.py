import tensorflow as tf
from numpy import *
from  utils_ import *

# create a network to train audio spectrograms
# model based on tf's cifar10 model: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
class audionet(Trainable):
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

    # This is the inference() method from cifar10 network in tensorflo
    def create(self, xdim, num_classes, final_layer ="prob"):

        """Build the CIFAR-10 model.
        Args:
          images: Images returned from distorted_inputs() or inputs().
        Returns:
          Logits.
        """
        # We instantiate all variables using tf.get_variable() instead of
        # tf.Variable() in order to share variables across multiple GPU training runs.
        # If we only ran this model on a single GPU, we could simplify this function
        # by replacing all instances of tf.get_variable() with tf.Variable().
        #
        # conv1
        input_tensors = tf.placeholder(tf.float32, xdim)
        batch_size = tf.shape(input_tensors)[0]
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, 3, 64],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(input_tensors, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(initial_value=tf.constant_initializer(0.0), shape=[64],name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, 64, 64],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(name='biases', shape= [64],initial_value=tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # local3
        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                                  stddev=0.04, wd=0.004)
            biases = tf.Variable(name='biases',shape= [384], initial_value=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # local4
        with tf.variable_scope('local4') as scope:
            weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                                  stddev=0.04, wd=0.004)
            biases =tf.Variable(name ='biases', shape = [192], initial_value=tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

        # linear layer(WX + b),
        # We don't apply softmax here because
        # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
        # and performs the softmax internally for efficiency.
        with tf.variable_scope('softmax_linear') as scope:
            weights = _variable_with_weight_decay('weights', [192, num_classes],
                                                  stddev=1 / 192.0, wd=0.0)
            biases =tf.Variable(name='biases',shape= [num_classes],
                                      initial_value= tf.constant_initializer(0.0))
            logits = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

        return logits



def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var =tf.Variable(name=name, shape=shape, initial_value= tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


