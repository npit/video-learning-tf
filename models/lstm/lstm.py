import tensorflow as tf
import tflearn


def define_lstm(inputTensor):
    num_hidden = 2  # number of hidden layers
    batchsize = 2  # number of data instances in incoming ndarray
    sequence_len = 16
    num_classes = 101
    input_dim=2
    state = None  # state vector


    cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden,state_is_tuple=True)
    # Initial state of the LSTM memory.
    state = tf.zeros([batchsize, cell.state_size[0]])
    # IO placeholders
    # x = tf.placeholder(tf.float32, [None, input_dim])
    # tf.reshape
    # x = [ x for _ in range(sequence_len)]

    # y = tf.placeholder(tf.float32, [None, num_classes])
    # execution
    # dynamic is useful only when the input is different than 16
    #output, state = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    print(inputTensor.shape)
    listTensor = tf.split(inputTensor,sequence_len,axis=0)

    output, state = tf.contrib.rnn.static_rnn(cell, listTensor, dtype=tf.float32)
    predictions = tf.nn.softmax(output)
    # val = tf.transpose(output, [1, 0, 2])
    # last = tf.gather(val, int(val.get_shape()[0]) - 1)
    # pass activations to a softmax
    # activation should be num_classes-dimensional vectors

    return predictions




#Lstm = define_lstm()
