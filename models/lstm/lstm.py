import tensorflow as tf
import tflearn

# loads of TODO, shameful display
def define(inputTensor, batch_size, num_classes):
    with tf.name_scope("LSTM"):

        # 1 )
        # check out donahue's caffe code to get ideas
        # https://people.eecs.berkeley.edu/~lisa_anne/LRCN_video_weights.html

        # 2 )
        #this SO post says to reformat the inputs to match the lstm dim
        # http://stackoverflow.com/questions/35056909/input-to-lstm-network-tensorflow

        # 3 )
        # useful: !!!
        # check this SO post for IO, shapes try to duplicate
        # http://stackoverflow.com/questions/39324520/understanding-tensorflow-lstm-input-shape

        # note that batch size predominantly is about the # of instances you will use to perform a weight update

        # num hidden neurons : the size of the hidden state ... vector, essentially.
        num_hidden = 256  # number of hidden neurons for the lstm layer. like dim of a fc layer. Not necessary to be eq. to number
        # of classes. for ex., if input size is vectors sized d , internal weight would be d x num_hidden


        sequence_len = 16 # specifies sequence len for a weight update to take place. after that, we reset the state
        # the above perhaps should be handled on the training loop. prolly. This implies that sequence length has to be <= the batch size

        input_dim=int(inputTensor.shape[-1])
        state = None  # state vector

        # LSTM basic cell
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden,state_is_tuple=True)
        # Initial memory state is blank
        #state = cell.zero_state(batchsize,tf.float32)
        #state  = tf.zeros([batchsize, cell.state_size[0]])


        # input tensor is a unstructured image ndarray
        # we want to process num_frames images, get the output and restore the state
        # expected tensor is of shape [batch, sequence_len, ... ]
        print(inputTensor.shape)



        ## static rnn case, where input is a lilst of tensors
        #listOfTensors = tf.split(inputTensor,(sequence_len,),axis=0,name = "split_rnn_input")
        #print(listOfTensors)
        #print(listOfTensors[0].shape)
        #output, state = tf.contrib.rnn.static_rnn(cell, listOfTensors, sequence_length = [sequence_len] * batchsize, dtype=tf.float32)

        # dynamic rnn case, where input is a signle tensor
        inputTensor = tf.reshape(inputTensor,(-1,sequence_len,input_dim),name="lstm_input_reshape")
        output, state = tf.nn.dynamic_rnn(cell, inputTensor, sequence_length=[sequence_len] * batch_size, dtype=tf.float32)
        print(output.shape)
        # get the output of the last time step only
        output_last_timestep = tf.slice(output,[0,sequence_len-1,0],[batch_size,1,num_hidden],name="lstm_output_reshape")
        output_last_timestep = tf.squeeze(output_last_timestep, axis=1,name="lstm_output_squeeze")
        print(output_last_timestep.shape)

        # for eg 2 layers, use
        #  cell = rnn_cell.MultiRNNCell([lstm_cell] * 2)


        # add a final fc layer to convert from num_hidden to num_classes

        fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
        fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

        fc_out_w = tf.Variable(fc_out__init, name="fc_out_w")
        fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")
        fc_out = tf.nn.xw_plus_b(output_last_timestep, fc_out_w, fc_out_b, name="fc_out")
        print(fc_out.shape)

    return fc_out




#Lstm = define_lstm()
