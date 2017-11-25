import tensorflow as tf
from utils_ import *
from tf_util import *
from defs_ import *

class lstm(Trainable):

    cell_varscope = "lstm_net_varscope"
    def make_cell(self, num_hidden, num_layers):
        '''
        Make the cell(s) object
        :param num_hidden:
        :param num_layers:
        :return:
        '''
        with tf.variable_scope(self.cell_varscope):
            cells = [tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                     for _ in range(num_layers)]
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        return cell

    def get_zero_state(self, batch_size, num_hidden, cells):
        """
        Creates and returs a zero state vector wrt to input cells
        :param num_hidden:
        :param batch_size:
        :param cells:
        :return:
        """
        zeros = tf.zeros([batch_size, num_hidden])
        zero_state = self.get_state_tuple(zeros, cells)
        return zero_state

    def get_state_tuple(self, state_vector, cells):
        """
        Get tuple state from a state vector
        :param state_vector:
        :param cells:
        :return:
        """
        state_tuple = tuple([tf.contrib.rnn.LSTMStateTuple(state_vector, state_vector) for _ in cells._cells])
        return state_tuple

    def manage_trainables(self, namescope_name):
        fc_vars = [f for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, namescope_name)
                   if namescope_name in f.name]
        cell_vars = [v for v in tf.trainable_variables() if v.name.startswith('rnn')]
        self.train_modified.extend(fc_vars + cell_vars)


    def apply_dropout(self, input_tensor, dropout_keep_prob):
        # add dropout
        if dropout_keep_prob > 0:
            output = tf.nn.dropout(input_tensor, keep_prob=dropout_keep_prob, name="lstm_dropout")
        else:
            output = input_tensor
        return output

    # basic, abstract lstm functions
    def forward_pass_sequence(self, input_tensor, input_state, input_dim, lstm_params, output_dim,
                              sequence_length, nonzero_sequence, dropout_prob=0.0, omit_output_fc = False):
        '''
        Pass an input sequence through the lstm, returning the output sequence state vector.
        :return: output and state
        '''

        info("LSTM structure [hidden, layers, fusion]: %s" % lstm_params)
        num_hidden, num_layers, fusion_method = lstm_params
        with tf.name_scope("lstm_net") as namescope:
            # define the cell(s)
            cells = self.make_cell(num_hidden, num_layers)

            if input_state is not None:
                # make input state conversion fc layer if necessary
                input_state = convert_dim_fc(input_state, num_hidden, name="input_state_fc")
            # evaluate it via dynamic_rnn
            output, state = self.evaluate_sequence(input_tensor, input_dim, cells, num_hidden, sequence_length, nonzero_sequence, input_state)

            if fusion_method != defs.fusion_method.state:
                # pool output batch to a vector
                output = apply_temporal_fusion(output, num_hidden, sequence_length, fusion_method)

                # add dropout
                output = self.apply_dropout(output, dropout_prob)

                if not omit_output_fc:
                    # map to match the output dimension
                    output = convert_dim_fc(output, output_dim, "output_fc")
            else:
                # consider the state as the fusion of the input - no need to process the produced output
                output = None


            # get trainable layers
            self.manage_trainables(namescope)

            return output, state


    def evaluate_sequence(self, inputTensor, input_dim, cells, num_hidden, sequence_len, nonzero_per_sequence=None, init_state=None):
        '''
        Evaluate a sequence through the lstm
        :param inputTensor: the tensor to be evaluated
        :param input_dim: the data vector length in the tensor
        :param cells: MultiRNNCell list
        :param sequence_len: The max sequence evaluat-able in the input tensor. Integer or list of integers
        :param num_hidden: The lstm state length
        :param nonzero_per_sequence: The non-zero (non-pad) elements per sequence in the batch, <= sequence_len
        :param init_state: The initial state vector
        :return: Output tensor and state vector
        '''

        # reshape input tensor from shape [ num_items * num_frames_per_item , input_dim ] to
        # [ num_items , num_frames_per_item , input_dim ]
        inputTensor = print_tensor(inputTensor, "inputTensor in lstm evaluate")
        sequence_len = print_tensor(sequence_len, "sequence_len in lstm evaluate")
        nonzero_per_sequence = print_tensor(nonzero_per_sequence, "nonzero_per_sequence in lstm evaluate")
        inputTensor = tf.reshape(inputTensor, [-1, sequence_len, input_dim], name="lstm_input_reshape")
        inputTensor = print_tensor(inputTensor, "input reshaped")

        # get the batch size during run.
        batch_size = tf.shape(inputTensor)[0]

        # get initial state
        if init_state is not None:
            if len(init_state.shape) == 1:
                init_state = tf.expand_dims(init_state, 0)
            init_state = self.get_state_tuple(init_state, cells)

        if nonzero_per_sequence is None:
            # all elements in the sequence are good2go
            # specify the sequence length for each batch item: [ numitemframes for i in range(batchsize)]

            _seq_len = tf.fill(tf.expand_dims(batch_size, 0), tf.constant(value=sequence_len, dtype=tf.int32))
        else:
            _seq_len = nonzero_per_sequence

        # forward pass through the network
        output, state = tf.nn.dynamic_rnn(cells, inputTensor, sequence_length=_seq_len, dtype=tf.float32,
                                          initial_state=init_state)
        return output, state

    def generate_feedback_sequence(self, input_tensors, batch_size, output_dim, sequence_length, num_hidden, num_layers,
                                   start_vector_arg, embedding_matrix_arg, visual_input_mode, return_type = defs.return_type.argmax_index):
        """
        Process that consumes a single input and state vector pair, and feeds the i-th output to the i+1 input
        :param return_type: specify whether to return output tensors or just the argmax index
        :param input_tensors:
        :param state_dim:
        :param output_dim:
        :param sequence_length:
        :param num_hidden:
        :param num_layers:
        :param start_vector:
        :param end_vector:
        :param embedding_matrix:
        :return:
        """

        with tf.name_scope("lstm_net") as namescope:

            # make cells
            cells = self.make_cell(num_hidden, num_layers)

            input_dim = int(input_tensors.shape[-1])

            # make input tensors conversion fc layer, if necessary
            if input_tensors is not None:
                if visual_input_mode == defs.rnn_visual_mode.state_bias:
                    # tensor should match the state dimension
                    input_tensors = convert_dim_fc(input_tensors, num_hidden, name="input_state_fc")
                    # update the input dimension
                    input_dim = num_hidden

            if return_type == defs.return_type.argmax_index:
                index_accumulation = tf.Variable(initial_value=np.zeros([0], np.int64), dtype=tf.int64,
                                                 trainable=False,
                                                 name="index_accumulation")
                # no need to load it via checkpoint
                self.ignorable_variable_names.append(index_accumulation.name)
            elif return_type == defs.return_type.standard:
                vector_accumulation = tf.Variable(initial_value=np.zeros([0, output_dim], np.float32),
                                                  dtype=tf.float32,
                                                  trainable=False, name="vector_accumulation")
                state_accumulation = tf.Variable(initial_value=np.zeros([0, input_dim], np.float32),
                                                 dtype=tf.float32,
                                                 trainable=False, name="state_accumulation")
                # no need to load it via checkpoint
                self.ignorable_variable_names.extend([vector_accumulation.name, state_accumulation.name])
            else:
                error("Undefined lstm return type [%s]" % return_type)


            # make tf variable constants
            start_vector = tf.constant(start_vector_arg, tf.float32, [1, len(start_vector_arg)])
            embedding_matrix = tf.constant(embedding_matrix_arg, tf.float32)

            # outer loop on batch size
            for batch_index in range(batch_size):
                # slice input tensors, getting a vector per loop
                if input_tensors is not None:
                    input_vector = tf.slice(input_tensors, [batch_index, 0], [1, input_dim])

                # set "defaults", i.e. zero initial state and start_vector as the initial vector
                io_vector = start_vector
                io_state = self.get_zero_state(1, num_hidden, cells)

                debug("Making feedback loop for the %d-th batch item, with a max sequence length %d" % (1+batch_index, sequence_length))
                # inner loop on the sequence length
                for i in range(sequence_length):

                    # the <visual_input_mode> determines how to handle the input tensors
                    if visual_input_mode == defs.rnn_visual_mode.state_bias:
                        # the visual input is a initial state: set it at the first sequence, if provided
                        if i == 0:
                            if input_tensors is not None:
                                io_state = self.get_state_tuple(input_vector, cells)

                    elif visual_input_mode == defs.rnn_visual_mode.input_concat:
                        # the visual input should be concatenated with the timestep input
                        io_vector = tf.concat([io_vector, input_vector], axis=1)

                    elif visual_input_mode == defs.rnn_visual_mode.input_bias:
                        # the visual input is an initial input: set it as first timestep input, prior to start_vector
                        if i==0:
                            io_vector = input_vector
                        elif i == 1:
                            # the 2nd sequence element is the actual start vector
                            io_vector = start_vector
                    else:
                        error("Undefined rnn visual input mode [%s]" % visual_input_mode)

                    # evaluate
                    if i > 0 : tf.get_variable_scope().reuse_variables()

                    io_vector, io_state = cells(io_vector, io_state, scope="rnn/multi_rnn_cell")

                    io_vector = convert_dim_fc(io_vector, output_dim, "output_fc", reuse=i>0)

                    # for a description tasks, get the corresponding word vector
                    io_vector, word_index = self.get_embedding_from_logits(io_vector, embedding_matrix)

                    if return_type == defs.return_type.argmax_index:
                        # for input bias mode, no need to store the first element
                        if not (visual_input_mode == defs.rnn_visual_mode.input_bias and i == 0):
                            index_accumulation = tf.concat([index_accumulation,word_index],0)
                    else:
                        # for input bias mode, no need to store the first element
                        if not (visual_input_mode == defs.rnn_visual_mode.input_bias and i == 0):
                            vector_accumulation = tf.concat([vector_accumulation, io_vector],0)
                            state_accumulation = tf.concat([state_accumulation, io_state],0)
            return index_accumulation

    def get_embedding_from_logits(self, logits, embedding_matrix):
        """

        :param logits:
        :param embedding_matrix:
        :return:
        """
        argmax = tf.arg_max(logits, 1)
        embedding = tf.gather(embedding_matrix, argmax)
        return embedding, argmax





    output = None

    def define_encoder(self,input_tensor, dataset, settings):
        with tf.name_scope("LSTM_encoder") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_hidden = settings.lstm_num_hidden
            sequence_len = dataset.num_frames_per_clip

            # LSTM basic cell
            with tf.variable_scope("LSTM_ac_vs") as varscope:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                cell_vars = [v for v in tf.all_variables()
                             if v.name.startswith(varscope.name)]
                self.train_modified.extend(cell_vars)
            debug("LSTM input : %s" % str(input_tensor.shape))
            # get LSTM rawoutput
            _, state = self.rnn_dynamic(input_tensor, cell, sequence_len, num_hidden)
            debug("LSTM state output : %s" % str(state.shape))
            self.output = state

    # input
    def define_decoder(self,input_state, input_words,  dataset, settings):
        with tf.name_scope("LSTM_decoder") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_hidden = settings.lstm_num_hidden
            sequence_len = dataset.num_frames_per_clip
            dropout_keep_prob = settings.dropout_keep_prob

            # LSTM basic cell
            with tf.variable_scope("LSTM_decoder_vs") as varscope:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                cell_vars = [v for v in tf.all_variables()
                             if v.name.startswith(varscope.name)]
                self.train_modified.extend(cell_vars)
            debug("LSTM input : %s" % str(input_words.shape))
            # get LSTM rawoutput
            sequence_data, state = self.rnn_dynamic(input_words, cell, sequence_len, num_hidden, settings.logging_level,init_state=input_state)
            debug("LSTM state output : %s" % str(state.shape))
            self.output = sequence_data


    def define_activity_recognition(self,inputTensor, dataset, settings):
        with tf.name_scope("LSTM_ac") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_hidden = settings.lstm_num_hidden
            num_classes = dataset.num_classes
            sequence_len = dataset.num_frames_per_clip
            frame_fusion_type = settings.frame_fusion
            dropout_keep_prob = settings.dropout_keep_prob

            # LSTM basic cell
            with tf.variable_scope("LSTM_ac_vs") as varscope:

                cells = [tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                         for _ in range(settings.lstm_num_layers)]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            debug("LSTM input : %s" % str(inputTensor.shape))

            # get LSTM rawoutput
            output, _ = self.rnn_dynamic(inputTensor,cell,sequence_len, num_hidden)
            debug("LSTM raw output : %s" % str(output.shape))

            if frame_fusion_type == defs.fusion_method.last:
                # keep only the response at the last time step
                output = tf.slice(output,[0,sequence_len-1,0],[-1,1,num_hidden],name="lstm_output_reshape")
                debug("LSTM last timestep output : %s" % str(output.shape))
                # squeeze empty dimension to get vector
                output = tf.squeeze(output, axis=1, name="lstm_output_squeeze")
                debug("LSTM squeezed output : %s" % str(output.shape))

            elif frame_fusion_type == defs.fusion_method.avg:
                # average per-timestep results
                output = tf.reduce_mean(output, axis=1)
                debug("LSTM time-averaged output : %s" % str(output.shape))
            else:
                error("Undefined frame fusion type : %d" % frame_fusion_type)


            # add dropout
            if settings.do_training:
                output = tf.nn.dropout(output, keep_prob=dropout_keep_prob,name="lstm_dropout")

            # add a final fc layer to convert from num_hidden to a num_classes output
            # layer initializations
            fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
            fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

            # create the layers
            fc_out_w = tf.Variable(fc_out__init, name="fc_out_w")
            fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")
            self.output = tf.nn.xw_plus_b(output, fc_out_w, fc_out_b, name="fc_out")
            debug("LSTM final output : %s" % str(self.output.shape))

            # sort out trained vars
            fc_vars = [f for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, namescope)
                       if namescope in f.name]
            cell_vars = [v for v in tf.global_variables() if v.name.startswith(varscope.name)]
            self.train_modified.extend(fc_vars)
            self.train_modified.extend(cell_vars)





    def get_output(self):
        return self.output


    def define_imgdesc_inputstep_validation(self, inputTensor, image_vector_dim, captions_per_item, dataset, settings):

        with tf.name_scope("LSTM_id") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_classes = dataset.num_classes
            num_hidden = settings.lstm_num_hidden
            sequence_len = dataset.max_caption_length + 1  # plus one for the BOS
            info("Sequence length %d" % sequence_len)
            # LSTM basic cell
            with tf.variable_scope("LSTM_id_vs") as varscope:

                # create the cell and fc variables
                cells = [tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                         for _ in range(settings.lstm_num_layers)]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                # layer initializations
                # add a final fc layer to convert from num_hidden to a num_classes output
                fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
                fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

                # create the layers
                fc_out_w = tf.Variable(fc_out__init, name="fc_out_w",)
                fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")

                debug("LSTM input : %s" % str(inputTensor.shape))

                # in validation mode, we need to have the word embeddings loaded up
                embedding_matrix = tf.constant(dataset.embedding_matrix,tf.float32)
                debug("Loaded the graph embedding matrix : %s" % embedding_matrix.shape)

                # batch_size =  tf.shape(inputTensor)[0]
                # item_index = tf.Variable(-1, tf.int32)
                # images_words_tensors = tf.split(inputTensor, dataset.batch_size_val, 0)

                predicted_words_for_batch = tf.Variable(np.zeros([0,sequence_len], np.int64), tf.int64, name="predicted_words_for_batch")
                empty_word_indexes = tf.Variable(np.zeros([0], np.int64), tf.int64, name="empty_word_indexes")
                # assumes fix-sized batches
                for item_index in range(dataset.batch_size_val):

                    # image_word_vector = print_tensor(image_word_vector ,"image_word_vector ")
                    # image_word_vector = inputTensor[item_index,:]
                    # image_word_vector = print_tensor(image_word_vector ,"image_word_vector ")

                    # item_index = tf.add(item_index , 1)
                    item_index = print_tensor(item_index ,"item_index ")

                    # image_vector = image_word_vector[0,:image_vector_dim]
                    image_vector = tf.slice(inputTensor,[item_index,0],[1,image_vector_dim])
                    image_vector = print_tensor(image_vector, "image_vector ")
                    # image_vector = print_tensor(image_vector ,"image_vector ")

                    # zero state vector for the initial state
                    current_state = tuple(tf.zeros([1,num_hidden], tf.float32) for _ in range(2))
                    output, current_state = cell(inputTensor[item_index:item_index+1,:], current_state, scope="rnn/basic_lstm_cell")

                    word_embedding, word_index = self.logits_to_word_vectors_tf(embedding_matrix, fc_out_w,
                                                                         fc_out_b, output, defs.caption_search.max)
                    word_embedding = print_tensor(word_embedding, "word_embedding 0")
                    # save predicted word index
                    current_word_indexes = tf.concat([empty_word_indexes, word_index],0)

                    for step in range(1,sequence_len):

                        input_vec = tf.concat([image_vector, word_embedding],axis=1)
                        input_vec = tf.squeeze(input_vec)
                        # idiotic fix of "input has to be 2D"
                        input_vec = tf.stack([input_vec ,input_vec ])

                        tf.get_variable_scope().reuse_variables()

                        logits, current_state = cell(input_vec[0:1,:], current_state, scope="rnn/basic_lstm_cell")
                        # debug("LSTM iteration #%d state : %s" % (step, [str(x.shape) for x in state]))
                        word_embedding, word_index = self.logits_to_word_vectors_tf(embedding_matrix,
                                                            fc_out_w, fc_out_b, logits, defs.caption_search.max)
                        # debug("LSTM iteration #%d output : %s" % (step, data_io.shape))
                        current_word_indexes = tf.concat([current_word_indexes, word_index], axis=0)
                        word_embedding =  print_tensor(word_embedding,"word_embedding %d" % step)

                    current_word_indexes = print_tensor(current_word_indexes,"current_word_indexes")
                    # done for the current image, append to batch results
                    predicted_words_for_batch = tf.concat([predicted_words_for_batch, tf.expand_dims(current_word_indexes,0)],0)
                self.output = predicted_words_for_batch



    def logits_to_word_vectors_tf(self, embedding_matrix, weights, biases, logits, strategy=defs.caption_search.max):
        if strategy == defs.caption_search.max:
            # here we should select a word from the output
            logits_on_num_words = tf.nn.xw_plus_b(logits, weights, biases)
            # debug("LSTM iteration #%d output : %s" % (step, data_io.shape))
            # here we should extract the predicted label from the output tensor. There are a variety of selecting that
            # we ll try the argmax here => get the most likely caption. ASSUME batchsize of 1
            # get the max index of the output logit vector
            word_index = tf.argmax(logits_on_num_words, 1)
            # get the word embedding of that index / word, which is now the new input
            word_vector = tf.gather(embedding_matrix, word_index)
            # debug("Vectors from logits ,iteration #%d  is : %s" % (step, data_io.shape))

            return word_vector, word_index



    def define_imgdesc_inputstep(self, inputTensor, composite_dimension, num_words_caption, dataset,
                                 settings):
        with tf.name_scope("LSTM_id") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_classes = dataset.num_classes
            num_hidden = settings.lstm_num_hidden
            sequence_len = dataset.max_caption_length + 1  # plus one for the BOS
            dropout_keep_prob = settings.dropout_keep_prob

            # LSTM basic cell
            with tf.variable_scope("LSTM_id_vs") as varscope:
                cells = [tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                         for _ in range(settings.lstm_num_layers)]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                debug("LSTM input : %s" % str(inputTensor.shape))

                # get LSTM rawoutput
                output, _ = self.rnn_dynamic(inputTensor, cell, sequence_len, num_hidden, settings.logging_level, num_words_caption)
                output = print_tensor(output, "lstm raw output:")
                debug("LSTM raw output : %s" % str(output.shape))

                # reshape to num_batches * sequence_len x num_hidden
                output = tf.reshape(output, [-1, num_hidden])
                debug("LSTM recombined output : %s" % str(output.shape))
                output = print_tensor(output, "lstm recombined output")

                # add dropout
                if settings.do_training:
                    output = tf.nn.dropout(output, keep_prob=dropout_keep_prob, name="lstm_dropout")

                # add a final fc layer to convert from num_hidden to a num_classes output
                # layer initializations
                fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
                fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

                # create the layers
                fc_out_w = tf.Variable(fc_out__init, name="fc_out_w")
                fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")
                self.output = tf.nn.xw_plus_b(output, fc_out_w, fc_out_b, name="fc_out")
                debug("LSTM final output : %s" % str(self.output.shape))
                self.output = print_tensor(self.output, "lstm fc output")

                fc_vars = [f for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, namescope)
                           if f.name.startswith(namescope)]
                self.train_modified.extend(fc_vars)

                # include a dummy variables, used in the validation network to enable loading
                predicted_words_for_batch = tf.Variable(np.zeros([0, sequence_len], np.int64), tf.int64,name="predicted_words_for_batch")
                empty_word_indexes = tf.Variable(np.zeros([0], np.int64), tf.int64, name="empty_word_indexes")

    # receives a numwords x embeddim tensor of caption words and a numimages x encodingdim tensor of images.
    def define_lstm_inputbias_loop(self, wordsTensor, biasTensor, num_words_per_caption, dataset, settings):
        with tf.name_scope("LSTM_id") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_classes = dataset.num_classes
            num_hidden = settings.lstm_num_hidden
            sequence_len = dataset.max_caption_length + 1  # plus one for the BOS
            dropout_keep_prob = settings.dropout_keep_prob
            info("Defining lstm with seqlen: %d" % (sequence_len))

            # add a final fc layer to convert from num_hidden to a num_classes output
            # layer initializations
            fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
            fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

            # create the layers
            fc_out_w = tf.Variable(fc_out__init, name="fc_out_w")
            fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")

            self.output = tf.Variable(initial_value=np.zeros([0, num_classes],np.float32), dtype=tf.float32,trainable=False,name="LSTM_output")
            # need to map the image to the state dimension. If not equal, add an fc layer transformation
            bias_dimension = int(biasTensor.shape[1])
            if bias_dimension != num_hidden:
                info("Mapping visual bias %d-layer to the %d-sized LSTM state." % (
                bias_dimension, num_hidden))
                # layer initializations
                fc_bias_state_w_init = tf.truncated_normal((bias_dimension, num_hidden), stddev=0.05,  name="fc_bias_state_w_init")
                fc_bias_state_b_init = tf.constant(0.1, shape=(num_hidden,), name="fc_bias_state_b_init")

                # create the layers
                fc_bias_state_w = tf.Variable(fc_bias_state_w_init, name="fc_bias_state_w")
                fc_bias_state_b = tf.Variable(fc_bias_state_b_init, name="fc_bias_state_b")

                biasTensor = tf.nn.xw_plus_b(biasTensor, fc_bias_state_w, fc_bias_state_b, name="fc_out")

            # probably has to be done with a for on each visual input ...

            # LSTM basic cell
            with tf.variable_scope("LSTM_id_vs") as varscope:
                cells = [tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                         for _ in range(settings.lstm_num_layers)]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

                debug("LSTM input : %s" % str(wordsTensor.shape))
                debug("LSTM input state bias : %s" % str(biasTensor.shape))
                wordsTensor = print_tensor(wordsTensor,"total words tensor")
                biasTensor = print_tensor(biasTensor,"total bias tensor")

                for image_index in range(dataset.batch_size_train):
                    if image_index > 0:
                        tf.get_variable_scope().reuse_variables()

                    bias_vector = biasTensor[image_index]
                    words_vectors = wordsTensor[image_index * sequence_len : (1+image_index) * sequence_len,:]

                    bias_vector = print_tensor(bias_vector, "bias vector")
                    words_vectors = print_tensor(words_vectors, "word vectors")
                    num_words_in_caption = num_words_per_caption[image_index]
                    # get LSTM rawoutput
                    marginal_output, _ = self.rnn_dynamic(words_vectors, cell, sequence_len, num_hidden,
                                                          settings.logging_level, num_words_in_caption, bias_vector)
                    marginal_output = print_tensor(marginal_output, "lstm raw output:")
                    debug("LSTM raw output : %s" % str(marginal_output.shape))

                # reshape to num_batches * sequence_len x num_hidden
                    marginal_output = tf.reshape(marginal_output, [-1, num_hidden])
                    debug("LSTM recombined output : %s" % str(marginal_output.shape))
                    marginal_output = print_tensor(marginal_output, "lstm recombined output")

                    # add dropout
                    if settings.do_training:
                        marginal_output = tf.nn.dropout(marginal_output, keep_prob=dropout_keep_prob, name="lstm_dropout")

                    marginal_output = tf.nn.xw_plus_b(marginal_output, fc_out_w, fc_out_b, name="fc_out")

                    marginal_output = print_tensor(marginal_output, "lstm fc output")
                    self.output = tf.concat([self.output, marginal_output],axis=0)

                debug("LSTM final output : %s" % str(self.output.shape))

                # include a dummy variables, used in the validation network to enable loading

                predicted_words_for_batch = tf.Variable(initial_value=np.zeros([0, sequence_len], np.int64), dtype=tf.int64, name="predicted_words_for_batch",trainable=False)
                empty_word_indexes = tf.Variable(initial_value=np.zeros([0], np.int64), dtype=tf.int64,name="empty_word_indexes",trainable=False)

                # sort out trained vars
                fc_vars = [f for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, namescope)
                           if namescope in  f.name]
                cell_vars = [v for v in tf.global_variables() if v.name.startswith(varscope.name)]
                self.train_modified.extend(fc_vars)
                self.train_modified.extend(cell_vars)






                # receives a numwords x embeddim tensor of caption words and a numimages x encodingdim tensor of images.

    def define_lstm_inputbias_loop_validation(self, biasTensor, num_words_per_caption, dataset, settings):
        with tf.name_scope("LSTM_id") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_classes = dataset.num_classes
            num_hidden = settings.lstm_num_hidden
            sequence_len = dataset.max_caption_length + 1  # plus one for the BOS
            info("Defining lstm with seqlen: %d" % (sequence_len))

            # add a final fc layer to convert from num_hidden to a num_classes output
            # layer initializations
            fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
            fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

            # create the layers
            fc_out_w = tf.Variable(fc_out__init, name="fc_out_w")
            fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")



            # need to map the image to the state dimension. If not equal, add an fc layer transformation
            bias_dimension = int(biasTensor.shape[1])
            if bias_dimension != num_hidden:
                info("Mapping visual bias %d-layer to the %d-sized LSTM state." % (
                    bias_dimension, num_hidden))
                # layer initializations
                fc_bias_state_w_init = tf.truncated_normal((bias_dimension, num_hidden), stddev=0.05,
                                                           name="fc_bias_state_w_init")
                fc_bias_state_b_init = tf.constant(0.1, shape=(num_hidden,), name="fc_bias_state_b_init")

                # create the layers
                fc_bias_state_w = tf.Variable(fc_bias_state_w_init, name="fc_bias_state_w")
                fc_bias_state_b = tf.Variable(fc_bias_state_b_init, name="fc_bias_state_b")

                biasTensor = tf.nn.xw_plus_b(biasTensor, fc_bias_state_w, fc_bias_state_b, name="fc_out")

            # probably has to be done with a for on each visual input ...


            # LSTM basic cell
            with tf.variable_scope("LSTM_id_vs") as varscope:
                cells = [tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                         for _ in range(settings.lstm_num_layers)]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)


                predicted_words_for_batch = tf.Variable(np.zeros([0, sequence_len], np.int64), dtype=tf.int64,
                                                        name="predicted_words_for_batch", trainable=False)
                empty_word_indexes = tf.Variable(np.zeros([0], np.int64), tf.int64, name="empty_word_indexes")

                debug("LSTM input state bias : %s" % str(biasTensor.shape))

                biasTensor = print_tensor(biasTensor, "total bias tensor")

                for image_index in range(dataset.batch_size_val):
                    if image_index > 0:
                        tf.get_variable_scope().reuse_variables()

                    bias_vector = tf.expand_dims(biasTensor[image_index,:],0)
                    bias_vector = print_tensor(bias_vector , "bias vector")

                    # get the BOS embedding
                    bos_vector = tf.expand_dims(tf.constant(dataset.embedding_matrix[dataset.vocabulary.index('BOS'),:],tf.float32),0)

                    # zero state vector for the initial state

                    current_state = [tf.contrib.rnn.LSTMStateTuple(bias_vector, bias_vector) for _ in range(settings.lstm_num_layers)]
                    output, current_state = cell(bos_vector, current_state,scope="rnn/multi_rnn_cell")

                    word_embedding, word_index = self.logits_to_word_vectors_tf(dataset.embedding_matrix, fc_out_w,
                                                                                fc_out_b, output,
                                                                                defs.caption_search.max)
                    word_embedding = print_tensor(word_embedding, "word_embedding 0")
                    # save predicted word index
                    current_word_indexes = tf.concat([empty_word_indexes, word_index], 0)

                    for step in range(1, sequence_len):

                        tf.get_variable_scope().reuse_variables()

                        logits, current_state = cell(word_embedding, current_state, scope="rnn/multi_rnn_cell")
                        # debug("LSTM iteration #%d state : %s" % (step, [str(x.shape) for x in state]))
                        word_embedding, word_index = self.logits_to_word_vectors_tf(dataset.embedding_matrix,
                                                                                    fc_out_w, fc_out_b, logits,
                                                                                    defs.caption_search.max)
                        # debug("LSTM iteration #%d output : %s" % (step, data_io.shape))
                        current_word_indexes = tf.concat([current_word_indexes, word_index], axis=0)
                        word_embedding = print_tensor(word_embedding, "word_embedding %d" % step,
                                                      settings.logging_level)

                    current_word_indexes = print_tensor(current_word_indexes, "current_word_indexes",
                                                        settings.logging_level)
                    # done for the current image, append to batch results
                    predicted_words_for_batch = tf.concat(
                        [predicted_words_for_batch, tf.expand_dims(current_word_indexes, 0)], 0)
                self.output = predicted_words_for_batch






    # receives a numwords x embeddim tensor of caption words and a numimages x encodingdim tensor of images.
    def define_lstm_inputbias(self, wordsTensor, biasTensor, num_words_caption, dataset, settings):
        with tf.name_scope("LSTM_id") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_classes = dataset.num_classes
            num_hidden = settings.lstm_num_hidden
            sequence_len = dataset.max_caption_length + 1  # plus one for the BOS
            dropout_keep_prob = settings.dropout_keep_prob

            info("Bias tensor : %s" % str(biasTensor.shape))
            biasTensor = print_tensor(biasTensor,"bias tensor")

            # add a final fc layer to convert from num_hidden to a num_classes output
            # layer initializations
            fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
            fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

            # create the classification fc layer
            fc_out_w = tf.Variable(fc_out__init, name="fc_out_w")
            fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")

            # need to map the image to the state dimension. If not equal, add an fc layer transformation
            bias_dimension = int(biasTensor.shape[1])
            if bias_dimension != num_hidden:
                info("Mapping visual bias %d-layer to the %d-sized LSTM state." % (bias_dimension, num_hidden))
                # layer initializations
                fc_bias_state_w_init = tf.truncated_normal((bias_dimension, num_hidden), stddev=0.05, name="fc_bias_state_w_init")
                fc_bias_state_b_init = tf.constant(0.1, shape=(num_hidden,), name="fc_bias_state_b_init")

                # create the layers
                fc_bias_state_w = tf.Variable(fc_bias_state_w_init, name="fc_bias_state_w")
                fc_bias_state_b = tf.Variable(fc_bias_state_b_init, name="fc_bias_state_b")

                biasTensor = tf.nn.xw_plus_b(biasTensor, fc_bias_state_w, fc_bias_state_b, name="fc_out")

            info("Bias tensor post-mapping : %s" % str(biasTensor.shape))
            biasTensor = print_tensor(biasTensor, "bias tensor post mapping")
            # probably has to be done with a while on each visual input ...

            # LSTM basic cell
            with tf.variable_scope("LSTM_id_vs") as varscope:

                cells = [tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                         for _ in range(settings.lstm_num_layers)]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

                debug("LSTM input : %s" % str(wordsTensor.shape))
                debug("LSTM input state bias : %s" % str(biasTensor.shape))

                # get LSTM rawoutput
                output, _ = self.rnn_dynamic(wordsTensor, cell, sequence_len, num_hidden,
                                             settings.logging_level, num_words_caption, biasTensor)
                output = print_tensor(output, "lstm raw output:")
                debug("LSTM raw output : %s" % str(output.shape))

                # reshape to num_batches * sequence_len x num_hidden
                output = tf.reshape(output, [-1, num_hidden])
                debug("LSTM recombined output : %s" % str(output.shape))
                output = print_tensor(output, "lstm recombined output")

                # add dropout to the raw output
                if settings.do_training:
                    output = tf.nn.dropout(output, keep_prob=dropout_keep_prob, name="lstm_dropout")

                self.output = tf.nn.xw_plus_b(output, fc_out_w, fc_out_b, name="fc_out")
                debug("LSTM final output : %s" % str(self.output.shape))
                self.output = print_tensor(self.output, "lstm fc output")


                # sort out trained vars
                fc_vars = [f for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, namescope)
                           if namescope in  f.name]
                cell_vars = [v for v in tf.global_variables() if v.name.startswith(varscope.name)]
                self.train_modified.extend(fc_vars)
                self.train_modified.extend(cell_vars)

                # include a dummy variables, used in the validation network to enable loading
                predicted_words_for_batch = tf.Variable(initial_value=np.zeros([0, sequence_len], np.int64), dtype=tf.int64,
                                                        name="predicted_words_for_batch", trainable = False)
                empty_word_indexes = tf.Variable(initial_value=np.zeros([0], np.int64), dtype=tf.int64, name="empty_word_indexes", trainable =False)

    def define_lstm_inputbias_validation(self, biasTensor, captions_per_item, dataset, settings):

        with tf.name_scope("LSTM_id") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_classes = dataset.num_classes
            num_hidden = settings.lstm_num_hidden
            sequence_len = dataset.max_caption_length + 1  # plus one for the BOS
            info("Sequence length %d" % sequence_len)

            # need to map the image to the state dimension. If not equal, add an fc layer transformation
            bias_dimension = int(biasTensor.shape[1])
            if bias_dimension != num_hidden:
                info("Mapping visual bias %d-layer to the %d-sized LSTM state." % (bias_dimension, num_hidden))
                # layer initializations
                fc_bias_state_w_init = tf.truncated_normal((bias_dimension, num_hidden), stddev=0.05,
                                                           name="fc_bias_state_w_init")
                fc_bias_state_b_init = tf.constant(0.1, shape=(num_hidden,), name="fc_bias_state_b_init")

                # create the layers
                fc_bias_state_w = tf.Variable(fc_bias_state_w_init, name="fc_bias_state_w")
                fc_bias_state_b = tf.Variable(fc_bias_state_b_init, name="fc_bias_state_b")

                biasTensor = tf.nn.xw_plus_b(biasTensor, fc_bias_state_w, fc_bias_state_b, name="fc_out")

            # LSTM basic cell
            with tf.variable_scope("LSTM_id_vs") as varscope:

                # create the cell and fc variables
                cells = [tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                         for _ in range(settings.lstm_num_layers)]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                # layer initializations
                # add a final fc layer to convert from num_hidden to a num_classes output
                fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
                fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

                # create the layers
                fc_out_w = tf.Variable(fc_out__init, name="fc_out_w", )
                fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")

                debug("LSTM bias : %s" % str(biasTensor.shape))

                # in validation mode, we need to have the word embeddings loaded up
                embedding_matrix = tf.constant(dataset.embedding_matrix, tf.float32)
                debug("Loaded the graph embedding matrix : %s" % embedding_matrix.shape)

                # batch_size =  tf.shape(inputTensor)[0]
                # item_index = tf.Variable(-1, tf.int32)
                # images_words_tensors = tf.split(inputTensor, dataset.batch_size_val, 0)

                predicted_words_for_batch = tf.Variable(np.zeros([0, sequence_len], np.int64), tf.int64,
                                                        name="predicted_words_for_batch")
                empty_word_indexes = tf.Variable(np.zeros([0], np.int64), tf.int64, name="empty_word_indexes")
                # assumes fix-sized batches
                for item_index in range(dataset.batch_size_val):

                    # image_word_vector = print_tensor(image_word_vector ,"image_word_vector ")
                    # image_word_vector = inputTensor[item_index,:]
                    # image_word_vector = print_tensor(image_word_vector ,"image_word_vector ")
                    image_vector, inputTensor = None, None

                    # item_index = tf.add(item_index , 1)
                    item_index = print_tensor(item_index, "item_index ")

                    # image_vector = image_word_vector[0,:image_vector_dim]
                    bias_vector = biasTensor[item_index]
                    # image_vector = tf.slice(inputTensor, [item_index, 0], [1, image_vector_dim])
                    # image_vector = print_tensor(image_vector, "image_vector ")
                    # image_vector = print_tensor(image_vector ,"image_vector ")

                    # zero state vector for the initial state
                    current_state = tf.contrib.rnn.LSTMStateTuple(bias_vector , bias_vector )
                    output, current_state = cell(inputTensor[item_index:item_index + 1, :], current_state,
                                                 scope="rnn/basic_lstm_cell")

                    word_embedding, word_index = self.logits_to_word_vectors_tf(embedding_matrix,
                                                                                fc_out_w,
                                                                                fc_out_b, output,
                                                                                defs.caption_search.max)
                    word_embedding = print_tensor(word_embedding, "word_embedding 0")
                    # save predicted word index
                    current_word_indexes = tf.concat([empty_word_indexes, word_index], 0)

                    for step in range(1, sequence_len):
                        input_vec = tf.concat([image_vector, word_embedding], axis=1)
                        input_vec = tf.squeeze(input_vec)
                        # idiotic fix of "input has to be 2D"
                        input_vec = tf.stack([input_vec, input_vec])

                        tf.get_variable_scope().reuse_variables()

                        logits, current_state = cell(input_vec[0:1, :], current_state,
                                                     scope="rnn/basic_lstm_cell")
                        # debug("LSTM iteration #%d state : %s" % (step, [str(x.shape) for x in state]))
                        word_embedding, word_index = self.logits_to_word_vectors_tf(embedding_matrix,
                                                                                    fc_out_w, fc_out_b, logits,
                                                                                    defs.caption_search.max)
                        # debug("LSTM iteration #%d output : %s" % (step, data_io.shape))
                        current_word_indexes = tf.concat([current_word_indexes, word_index], axis=0)
                        word_embedding = print_tensor(word_embedding, "word_embedding %d" % step)

                    current_word_indexes = print_tensor(current_word_indexes, "current_word_indexes")
                    # done for the current image, append to batch results
                    predicted_words_for_batch = tf.concat(
                        [predicted_words_for_batch, tf.expand_dims(current_word_indexes, 0)], 0)
                self.output = predicted_words_for_batch


    ## dynamic rnn case, where input is a single tensor
    def rnn_dynamic(self,inputTensor, cell, sequence_len, num_hidden, elements_per_sequence = None, init_state=None):
        # data vector dimension
        input_dim = int(inputTensor.shape[-1])

        # reshape input tensor from shape [ num_videos * num_frames_per_vid , input_dim ] to
        # [ num_videos , num_frames_per_vid , input_dim ]
        inputTensor = print_tensor(inputTensor ,"inputTensor  in rnn_dynamic")
        inputTensor = tf.reshape(inputTensor, (-1, sequence_len, input_dim), name="lstm_input_reshape")
        inputTensor = print_tensor(inputTensor, "input reshaped")

        # get the batch size during run. Make zero state to 2 - tuple of [batch_size, num_hidden]
        # 2-tuple state due to the sate_is_tuple LSTM cell
        batch_size = tf.shape(inputTensor)[0]
        if init_state is None:
            zero_state = [tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, num_hidden]),tf.zeros([batch_size, num_hidden])) \
                for _ in cell._cells]
        else:
            if len(init_state.shape) == 1:
                init_state = tf.expand_dims(init_state,0)
            zero_state = [tf.contrib.rnn.LSTMStateTuple(init_state, init_state) for _ in cell._cells]

        zero_state = tuple(zero_state)
        if elements_per_sequence is None:
            # all elements in the sequence are good2go
            # specify the sequence length for each batch item: [ numvideoframes for i in range(batchsize)]
            _seq_len = tf.fill(tf.expand_dims(batch_size, 0), tf.constant(sequence_len, dtype=tf.int64))
        else:
            _seq_len = elements_per_sequence

        # forward pass through the network
        output, state = tf.nn.dynamic_rnn(cell, inputTensor, sequence_length=_seq_len, dtype=tf.float32,
                                          initial_state=zero_state)
        return output, state




