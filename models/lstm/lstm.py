import tensorflow as tf
import tflearn

# loads of TODO, shameful display
def define(inputTensor, num_classes):
    with tf.name_scope("LSTM"):
        num_hidden = 32  # number of hidden neurons for the lstm layer. like dim of a fc layer. Not necessary to be eq. to number
        # of classes. for ex., if input size is vectors sized d , internal weight would be d x num_hidden
        batchsize = 2  # number of data instances in incoming ndarray
        sequence_len = 16 # specifies sequence len for a weight update to take place. after that, we reset the state
        # the above perhaps should be handled on the training loop. prolly. This implies that sequence length has to be <= the batch size
        num_classes = 101
        input_dim=4096
        state = None  # state vector

        # LSTM basic cell
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden,state_is_tuple=True)
        # Initial memory state is blank
        state = cell.zero_state(batchsize,tf.float32)
        #state  = tf.zeros([batchsize, cell.state_size[0]])


        # input tensor is a unstructured image ndarray
        # we want to process num_frames images, get the output and restore the state
        # expected tensor is of shape [batch, sequence_len, ... ]
        print(inputTensor.shape)
        #inputTensor = tf.reshape(inputTensor,(-1,sequence_len,input_dim))
        listOfTensors = tf.split(inputTensor,(sequence_len,),axis=0,name = "split_rnn_input")
        print(listOfTensors)
        print(listOfTensors[0].shape)
        output, state = tf.contrib.rnn.static_rnn(cell, listOfTensors, dtype=tf.float32)

        # for eg 2 layers, use
        #  cell = rnn_cell.MultiRNNCell([lstm_cell] * 2)


        # add a final fc layer to convert from num_hidden to num_classes



    return output




#Lstm = define_lstm()
