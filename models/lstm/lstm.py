import tensorflow as tf
import tflearn

# loads of TODO
def define(inputTensor, num_classes):
    with tf.name_scope("LSTM") as scope:
        num_hidden = 101  # number of hidden neurons for the lstm layer. like dim of a fc layer
        batchsize = 2  # number of data instances in incoming ndarray
        sequence_len = 16 # specifies sequence len for a weight update to take place. after that, we reset the state
        # the above perhaps should be handled on the training loop. prolly. This implies that sequence length has to be <= the batch size
        num_classes = 101
        input_dim=2
        state = None  # state vector


        cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden,state_is_tuple=True,name="lstm_cell")
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
        listTensor = tf.split(inputTensor,sequence_len,axis=0,name = "split_rnn_input")

        output, state = tf.contrib.rnn.static_rnn(cell, listTensor, dtype=tf.float32,name="static_rnn_run")

        # add a final fc layer to convert from num_hidden to num_classes

        # val = tf.transpose(output, [1, 0, 2])
        # last = tf.gather(val, int(val.get_shape()[0]) - 1)
        # pass activations to a softmax
        # activation should be num_classes-dimensional vectors

    return output




#Lstm = define_lstm()
