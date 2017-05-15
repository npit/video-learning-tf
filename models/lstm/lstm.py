import tensorflow as tf
import tflearn

# loads of TODO, shameful display
def define(inputTensor, input_batch_size, num_classes, keep_prob = 0.5):
    with tf.name_scope("LSTM"):
        print(" TODO create train and test phases in network, return 2 ops for each ? ?? " )

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



        # static rnn case, where input is a lilst of tensors
        #############################################################
        #
        # use : http://stackoverflow.com/questions/42520418/how-to-multiply-list-of-tensors-by-single-tensor-on-tensorflow
        # get a list of sequence_length size, where each tensor is of size batchsize x feature_dim
        # listOfTensors = tf.split(inputTensor,sequence_len,axis=0,name = "split_lstm_input")
        # print(len(listOfTensors))
        # print(listOfTensors)
        # print(listOfTensors[0].shape)
        # output, state = tf.contrib.rnn.static_rnn(cell, listOfTensors, dtype=tf.float32)
        # output = tf.stack(output,axis=1,name="lstm_output_stack")


        # dynamic rnn case, where input is a single tensor
        #############################################################

        inputTensor = tf.reshape(inputTensor,(-1,sequence_len,input_dim),name="lstm_input_reshape")

        batch_size = tf.shape(inputTensor)[0]
        _seq_len = tf.fill(tf.expand_dims(batch_size, 0),
                           tf.constant(sequence_len, dtype=tf.int64))
        output, state = tf.nn.dynamic_rnn(cell, inputTensor, sequence_length=_seq_len , dtype=tf.float32)



        print(output.shape)

        output = tf.slice(output,[0,sequence_len-1,0],[-1,1,num_hidden],name="lstm_output_reshape")

        output = tf.squeeze(output, axis=1, name="lstm_output_squeeze")

        print(output.shape)

        # for eg 2 layers, use
        #  cell = rnn_cell.MultiRNNCell([lstm_cell] * 2)

        # add dropout
        output = tf.nn.dropout(output, keep_prob=keep_prob,name="lstm_dropout")

        # add a final fc layer to convert from num_hidden to num_classes

        fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
        fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

        fc_out_w = tf.Variable(fc_out__init, name="fc_out_w")
        fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")
        fc_out = tf.nn.xw_plus_b(output, fc_out_w, fc_out_b, name="fc_out")
        print(fc_out.shape)

    return fc_out




#Lstm = define_lstm()
