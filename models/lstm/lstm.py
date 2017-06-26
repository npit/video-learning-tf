import tensorflow as tf
from utils_ import *

class lstm(Trainable):

    output = None

    def define_activity_recognition(self,inputTensor, dataset, settings):
        with tf.name_scope("LSTM_ac") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_hidden = settings.lstm_num_hidden
            num_classes = dataset.num_classes
            logger = dataset.logger
            sequence_len = dataset.num_frames_per_clip
            frame_pooling_type = settings.frame_pooling_type
            dropout_keep_prob = settings.dropout_keep_prob

            # LSTM basic cell
            with tf.variable_scope("LSTM_ac_vs") as varscope:

                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden,state_is_tuple=True)
                cell_vars = [v for v in tf.all_variables()
                             if v.name.startswith(varscope.name)]
                self.train_modified.extend(cell_vars)
            logger.debug("LSTM input : %s" % str(inputTensor.shape))

            # get LSTM rawoutput
            output = self.rnn_dynamic(inputTensor,cell,sequence_len, num_hidden, logger)
            logger.debug("LSTM raw output : %s" % str(output.shape))

            if frame_pooling_type == defs.pooling.last:
                # keep only the response at the last time step
                output = tf.slice(output,[0,sequence_len-1,0],[-1,1,num_hidden],name="lstm_output_reshape")
                logger.debug("LSTM last timestep output : %s" % str(output.shape))
                # squeeze empty dimension to get vector
                output = tf.squeeze(output, axis=1, name="lstm_output_squeeze")
                logger.debug("LSTM squeezed output : %s" % str(output.shape))

            elif frame_pooling_type == defs.pooling.avg:
                # average per-timestep results
                output = tf.reduce_mean(output, axis=1)
                logger.debug("LSTM time-averaged output : %s" % str(output.shape))
            else:
                logger.error("Undefined frame pooling type : %d" % frame_pooling_type)
                error("Undefined frame pooling type")


            # add dropout
            output = tf.nn.dropout(output, keep_prob=dropout_keep_prob,name="lstm_dropout")

            # add a final fc layer to convert from num_hidden to a num_classes output
            # layer initializations
            fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
            fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

            # create the layers
            fc_out_w = tf.Variable(fc_out__init, name="fc_out_w")
            fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")
            self.output = tf.nn.xw_plus_b(output, fc_out_w, fc_out_b, name="fc_out")
            logger.debug("LSTM final output : %s" % str(self.output.shape))


            fc_vars = [ f for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, namescope)
                        if f.name.startswith(namescope)]
            self.train_modified.extend(fc_vars)



    def get_output(self):
        return self.output


    def define_image_description(self,inputTensor, composite_dimension, vocab_len, num_words_caption, dataset,settings):

        with tf.name_scope("LSTM_id") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_classes = vocab_len
            num_hidden = settings.lstm_num_hidden
            logger = settings.logger
            sequence_len = dataset.num_frames_per_clip + 1 # plus one for the BOS
            frame_pooling_type = settings.frame_pooling_type
            dropout_keep_prob = settings.dropout_keep_prob

            # LSTM basic cell
            with tf.variable_scope("LSTM_id_vs") as varscope:
                # REUSE LSTM VARS BY: https://stackoverflow.com/questions/36941382/tensorflow-shared-variables-error-with-simple-lstm-network
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden,state_is_tuple=True)
                cell_vars = [v for v in tf.global_variables()
                             if v.name.startswith(varscope.name)]
                self.train_modified.extend(cell_vars)
            logger.debug("LSTM input : %s" % str(inputTensor.shape))


            # check https://github.com/mosessoh/CNN-LSTM-Caption-Generator/blob/master/model.py
            # link is a simple image+word -> ... ?

            # neural talk trains like:
            # RNN training. The RNN is trained to combine a word (xt),
            # the previous context (ht−1) to predict the next word (yt).
            # We condition the RNN’s predictions on the image information
            # (bv) via bias interactions on the first step. The training
            # proceeds as follows (refer to Figure 4): We set h0 = ~0, x1 to
            # a special START vector, and the desired label y1 as the first
            # word in the sequence. Analogously, we set x2 to the word
            # vector of the first word and expect the network to predict
            # the second word, etc. Finally, on the last step when xT represents
            # the last word, the target label is set to a special END
            # token. The cost function is to maximize the log probability
            # assigned to the target labels (i.e. Softmax classifier).

            # so the input is the desired caption, shifted to the right once, and a BOS inserted at the start
            # the desired response is the caption, followed by an EOS

            # neural talk plugs in the image just as a state BIAS.
            # lrcn plugs in the image at the input, ADDED to the word embedding.
            # all should be well now.

            # so the implementation is as usual, I guess

            # initial state should be the BOS symbol


            # initial_state = tuple(tf.zeros([1,num_hidden], tf.float32) for _ in range(2))
            # for _ in range(sequence_len):
            #     output, state = cell(inputTensor, initial_state)
            #     initial_state = state
            #     inputTensor = output

            # get LSTM rawoutput
            output = self.rnn_dynamic(inputTensor,cell,sequence_len, num_hidden, logger,num_words_caption)
            output = print_tensor(output,"lstm raw output:")
            logger.debug("LSTM raw output : %s" % str(output.shape))

            # reshape to num_batches * sequence_len x num_hidden
            output = tf.reshape(output, [-1, num_hidden])
            logger.debug("LSTM recombined output : %s" % str(output.shape))
            output = print_tensor(output,"lstm recombined output")

            # add dropout
            output = tf.nn.dropout(output, keep_prob=dropout_keep_prob,name="lstm_dropout")

            # add a final fc layer to convert from num_hidden to a num_classes output
            # layer initializations
            fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
            fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

            # create the layers
            fc_out_w = tf.Variable(fc_out__init, name="fc_out_w")
            fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")
            self.output = tf.nn.xw_plus_b(output, fc_out_w, fc_out_b, name="fc_out")
            logger.debug("LSTM final output : %s" % str(self.output.shape))
            self.output = print_tensor(self.output, "lstm fc output")

            fc_vars = [ f for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, namescope)
                        if f.name.startswith(namescope)]
            self.train_modified.extend(fc_vars)




    ## dynamic rnn case, where input is a single tensor
    def rnn_dynamic(self,inputTensor, cell, sequence_len, num_hidden, logger, elements_per_sequence = None):
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

        if elements_per_sequence is None:
            # all elements in the sequence are good2go
            # specify the sequence length for each batch item: [ numvideoframes for i in range(batchsize)]
            _seq_len = tf.fill(tf.expand_dims(batch_size, 0), tf.constant(sequence_len, dtype=tf.int64))
        else:
            _seq_len = elements_per_sequence

        # forward pass through the network
        output, state = tf.nn.dynamic_rnn(cell, inputTensor, sequence_length=_seq_len, dtype=tf.float32,
                                          initial_state=zero_state)
        return output




