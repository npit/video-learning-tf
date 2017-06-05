import tensorflow as tf
from utils_ import print_tensor

def define(inputTensor, input_batch_size, num_classes, sequence_len,  logger, summaries, dropout_keep_prob = 0.5):
    with tf.name_scope("LSTM"):

        # num hidden neurons, the size of the hidden state vector
        num_hidden = 256

        # LSTM basic cell
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden,state_is_tuple=True)
        logger.debug("LSTM input : %s" % str(inputTensor.shape))

        # get LSTM rawoutput
        output = rnn_dynamic(inputTensor,cell,sequence_len, num_hidden, logger)
        logger.debug("LSTM raw output : %s" % str(output.shape))

        # keep only the response at the last time step
        output = tf.slice(output,[0,sequence_len-1,0],[-1,1,num_hidden],name="lstm_output_reshape")
        logger.debug("LSTM sliced output : %s" % str(output.shape))

        # squeeze empty dimension to get vector
        output = tf.squeeze(output, axis=1, name="lstm_output_squeeze")
        logger.debug("LSTM squeezed output : %s" % str(output.shape))

        # add dropout
        output = tf.nn.dropout(output, keep_prob=dropout_keep_prob,name="lstm_dropout")

        # add a final fc layer to convert from num_hidden to a num_classes output
        # layer initializations
        fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
        fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

        # create the layers
        fc_out_w = tf.Variable(fc_out__init, name="fc_out_w")
        fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")
        fc_out = tf.nn.xw_plus_b(output, fc_out_w, fc_out_b, name="fc_out")
        logger.debug("LSTM final output : %s" % str(fc_out.shape))

    return fc_out


## dynamic rnn case, where input is a single tensor
def rnn_dynamic(inputTensor, cell, sequence_len, num_hidden, logger):
    # data vector dimension
    input_dim = int(inputTensor.shape[-1])

    # reshape input tensor from shape [ num_videos * num_frames_per_vid , input_dim ] to
    # [ num_videos , num_frames_per_vid , input_dim ]
    inputTensor = tf.reshape(inputTensor, (-1, sequence_len, input_dim), name="lstm_input_reshape")
    logger.debug("reshaped inputTensor %s" % str(inputTensor.shape))

    # get the batch size during run. Make zero state to 2 - tuple of [batch_size, num_hidden]
    # 2-tuple state due to the sate_is_tuple LSTM cell
    batch_size = tf.shape(inputTensor)[0]
    zero_state = tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, num_hidden]),tf.zeros([batch_size, num_hidden]))

    # specify the sequence length for each batch item: [ numvideoframes for i in range(batchsize)]
    _seq_len = tf.fill(tf.expand_dims(batch_size, 0), tf.constant(sequence_len, dtype=tf.int64))
    # forward pass through the network
    output, state = tf.nn.dynamic_rnn(cell, inputTensor, sequence_length=_seq_len, dtype=tf.float32,
                                      initial_state=zero_state)
    return output


#Lstm = define_lstm()
