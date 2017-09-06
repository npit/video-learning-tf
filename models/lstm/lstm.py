import tensorflow as tf
from utils_ import *

class lstm(Trainable):


    output = None

    def define_encoder(self,input_tensor, dataset, settings):
        with tf.name_scope("LSTM_encoder") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_hidden = settings.lstm_num_hidden
            logger = dataset.logger
            sequence_len = dataset.num_frames_per_clip

            # LSTM basic cell
            with tf.variable_scope("LSTM_ac_vs") as varscope:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                cell_vars = [v for v in tf.all_variables()
                             if v.name.startswith(varscope.name)]
                self.train_modified.extend(cell_vars)
            logger.debug("LSTM input : %s" % str(input_tensor.shape))
            # get LSTM rawoutput
            _, state = self.rnn_dynamic(input_tensor, cell, sequence_len, num_hidden, logger, settings.logging_level)
            logger.debug("LSTM state output : %s" % str(state.shape))
            self.output = state

    # input
    def define_decoder(self,input_state, input_words,  dataset, settings):
        with tf.name_scope("LSTM_decoder") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_hidden = settings.lstm_num_hidden
            logger = dataset.logger
            sequence_len = dataset.num_frames_per_clip
            dropout_keep_prob = settings.dropout_keep_prob

            # LSTM basic cell
            with tf.variable_scope("LSTM_decoder_vs") as varscope:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                cell_vars = [v for v in tf.all_variables()
                             if v.name.startswith(varscope.name)]
                self.train_modified.extend(cell_vars)
            logger.debug("LSTM input : %s" % str(input_words.shape))
            # get LSTM rawoutput
            sequence_data, state = self.rnn_dynamic(input_words, cell, sequence_len, num_hidden, logger, settings.logging_level,init_state=input_state)
            logger.debug("LSTM state output : %s" % str(state.shape))
            self.output = sequence_data


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
            output, _ = self.rnn_dynamic(inputTensor,cell,sequence_len, num_hidden, logger,settings.logging_level)
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
                error("Undefined frame pooling type : %d" % frame_pooling_type, self.logger)


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
            logger.debug("LSTM final output : %s" % str(self.output.shape))


            fc_vars = [ f for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, namescope)
                        if f.name.startswith(namescope)]
            self.train_modified.extend(fc_vars)


    def get_output(self):
        return self.output


    def define_imgdesc_inputstep_validation(self, inputTensor, image_vector_dim, captions_per_item, dataset, settings):

        with tf.name_scope("LSTM_id") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_classes = dataset.num_classes
            num_hidden = settings.lstm_num_hidden
            logger = settings.logger
            sequence_len = dataset.max_caption_length + 1  # plus one for the BOS
            dataset.logger.info("Sequence length %d" % sequence_len)
            # LSTM basic cell
            with tf.variable_scope("LSTM_id_vs") as varscope:

                # create the cell and fc variables
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden,state_is_tuple=True)
                # layer initializations
                # add a final fc layer to convert from num_hidden to a num_classes output
                fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
                fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

                # create the layers
                fc_out_w = tf.Variable(fc_out__init, name="fc_out_w",)
                fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")

                logger.debug("LSTM input : %s" % str(inputTensor.shape))

                # in validation mode, we need to have the word embeddings loaded up
                embedding_matrix = tf.constant(dataset.embedding_matrix,tf.float32)
                logger.debug("Loaded the graph embedding matrix : %s" % embedding_matrix.shape)

                # batch_size =  tf.shape(inputTensor)[0]
                # item_index = tf.Variable(-1, tf.int32)
                # images_words_tensors = tf.split(inputTensor, dataset.batch_size_val, 0)

                predicted_words_for_batch = tf.Variable(np.zeros([0,sequence_len], np.int64), tf.int64, name="predicted_words_for_batch")
                empty_word_indexes = tf.Variable(np.zeros([0], np.int64), tf.int64, name="empty_word_indexes")
                # assumes fix-sized batches
                for item_index in range(dataset.batch_size_val):

                    # image_word_vector = print_tensor(image_word_vector ,"image_word_vector ",settings.logging_level)
                    # image_word_vector = inputTensor[item_index,:]
                    # image_word_vector = print_tensor(image_word_vector ,"image_word_vector ",settings.logging_level)

                    # item_index = tf.add(item_index , 1)
                    item_index = print_tensor(item_index ,"item_index ",settings.logging_level)

                    # image_vector = image_word_vector[0,:image_vector_dim]
                    image_vector = tf.slice(inputTensor,[item_index,0],[1,image_vector_dim])
                    image_vector = print_tensor(image_vector, "image_vector ", settings.logging_level)
                    # image_vector = print_tensor(image_vector ,"image_vector ",settings.logging_level)

                    # zero state vector for the initial state
                    current_state = tuple(tf.zeros([1,num_hidden], tf.float32) for _ in range(2))
                    output, current_state = cell(inputTensor[item_index:item_index+1,:], current_state, scope="rnn/basic_lstm_cell")

                    word_embedding, word_index = self.logits_to_word_vectors_tf(embedding_matrix, logger, fc_out_w,
                                                                         fc_out_b, output, defs.caption_search.max)
                    word_embedding = print_tensor(word_embedding, "word_embedding 0", settings.logging_level)
                    # save predicted word index
                    current_word_indexes = tf.concat([empty_word_indexes, word_index],0)

                    for step in range(1,sequence_len):

                        input_vec = tf.concat([image_vector, word_embedding],axis=1)
                        input_vec = tf.squeeze(input_vec)
                        # idiotic fix of "input has to be 2D"
                        input_vec = tf.stack([input_vec ,input_vec ])

                        tf.get_variable_scope().reuse_variables()

                        logits, current_state = cell(input_vec[0:1,:], current_state, scope="rnn/basic_lstm_cell")
                        # logger.debug("LSTM iteration #%d state : %s" % (step, [str(x.shape) for x in state]))
                        word_embedding, word_index = self.logits_to_word_vectors_tf(embedding_matrix, logger,
                                                            fc_out_w, fc_out_b, logits, defs.caption_search.max)
                        # logger.debug("LSTM iteration #%d output : %s" % (step, data_io.shape))
                        current_word_indexes = tf.concat([current_word_indexes, word_index], axis=0)
                        word_embedding =  print_tensor(word_embedding,"word_embedding %d" % step,settings.logging_level)

                    current_word_indexes = print_tensor(current_word_indexes,"current_word_indexes",settings.logging_level)
                    # done for the current image, append to batch results
                    predicted_words_for_batch = tf.concat([predicted_words_for_batch, tf.expand_dims(current_word_indexes,0)],0)
                self.output = predicted_words_for_batch



    def logits_to_word_vectors_tf(self, embedding_matrix, logger, weights, biases, logits, strategy=defs.caption_search.max):
        if strategy == defs.caption_search.max:
            # here we should select a word from the output
            logits_on_num_words = tf.nn.xw_plus_b(logits, weights, biases)
            # logger.debug("LSTM iteration #%d output : %s" % (step, data_io.shape))
            # here we should extract the predicted label from the output tensor. There are a variety of selecting that
            # we ll try the argmax here => get the most likely caption. ASSUME batchsize of 1
            # get the max index of the output logit vector
            word_index = tf.argmax(logits_on_num_words, 1)
            # get the word embedding of that index / word, which is now the new input
            word_vector = tf.gather(embedding_matrix, word_index)
            # logger.debug("Vectors from logits ,iteration #%d  is : %s" % (step, data_io.shape))

            return word_vector, word_index



    def define_imgdesc_inputstep(self, inputTensor, composite_dimension, num_words_caption, dataset,
                                 settings):
        with tf.name_scope("LSTM_id") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_classes = dataset.num_classes
            num_hidden = settings.lstm_num_hidden
            logger = settings.logger
            sequence_len = dataset.max_caption_length + 1  # plus one for the BOS
            dropout_keep_prob = settings.dropout_keep_prob

            # LSTM basic cell
            with tf.variable_scope("LSTM_id_vs") as varscope:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                cell_vars = [v for v in tf.global_variables()
                             if v.name.startswith(varscope.name)]
                self.train_modified.extend(cell_vars)
                logger.debug("LSTM input : %s" % str(inputTensor.shape))

                # get LSTM rawoutput
                output, _ = self.rnn_dynamic(inputTensor, cell, sequence_len, num_hidden, logger,settings.logging_level, num_words_caption)
                output = print_tensor(output, "lstm raw output:",settings.logging_level)
                logger.debug("LSTM raw output : %s" % str(output.shape))

                # reshape to num_batches * sequence_len x num_hidden
                output = tf.reshape(output, [-1, num_hidden])
                logger.debug("LSTM recombined output : %s" % str(output.shape))
                output = print_tensor(output, "lstm recombined output",settings.logging_level)

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
                logger.debug("LSTM final output : %s" % str(self.output.shape))
                self.output = print_tensor(self.output, "lstm fc output",settings.logging_level)

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
            logger = settings.logger
            sequence_len = dataset.max_caption_length + 1  # plus one for the BOS
            dropout_keep_prob = settings.dropout_keep_prob
            settings.logger.info("Defining lstm with seqlen: %d" % (sequence_len))

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
                logger.info("Mapping visual bias %d-layer to the %d-sized LSTM state." % (
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
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)

                logger.debug("LSTM input : %s" % str(wordsTensor.shape))
                logger.debug("LSTM input state bias : %s" % str(biasTensor.shape))
                wordsTensor = print_tensor(wordsTensor,"total words tensor",logging.DEBUG)
                biasTensor = print_tensor(biasTensor,"total bias tensor",logging.DEBUG)

                for image_index in range(dataset.batch_size_train):
                    if image_index > 0:
                        tf.get_variable_scope().reuse_variables()

                    bias_vector = biasTensor[image_index]
                    words_vectors = wordsTensor[image_index * sequence_len : (1+image_index) * sequence_len,:]

                    bias_vector = print_tensor(bias_vector, "bias vector", logging.DEBUG)
                    words_vectors = print_tensor(words_vectors, "word vectors", logging.DEBUG)
                    num_words_in_caption = num_words_per_caption[image_index]
                    # get LSTM rawoutput
                    marginal_output, _ = self.rnn_dynamic(words_vectors, cell, sequence_len, num_hidden, logger,
                                                          settings.logging_level, num_words_in_caption, bias_vector)
                    marginal_output = print_tensor(marginal_output, "lstm raw output:", settings.logging_level)
                    logger.debug("LSTM raw output : %s" % str(marginal_output.shape))

                # reshape to num_batches * sequence_len x num_hidden
                    marginal_output = tf.reshape(marginal_output, [-1, num_hidden])
                    logger.debug("LSTM recombined output : %s" % str(marginal_output.shape))
                    marginal_output = print_tensor(marginal_output, "lstm recombined output", settings.logging_level)

                    # add dropout
                    if settings.do_training:
                        marginal_output = tf.nn.dropout(marginal_output, keep_prob=dropout_keep_prob, name="lstm_dropout")

                    marginal_output = tf.nn.xw_plus_b(marginal_output, fc_out_w, fc_out_b, name="fc_out")

                    marginal_output = print_tensor(marginal_output, "lstm fc output", settings.logging_level)
                    self.output = tf.concat([self.output, marginal_output],axis=0)

                logger.debug("LSTM final output : %s" % str(self.output.shape))

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
            logger = settings.logger
            sequence_len = dataset.max_caption_length + 1  # plus one for the BOS
            dropout_keep_prob = settings.dropout_keep_prob
            settings.logger.info("Defining lstm with seqlen: %d" % (sequence_len))

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
                logger.info("Mapping visual bias %d-layer to the %d-sized LSTM state." % (
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
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                cell_vars = [v for v in tf.global_variables()
                             if v.name.startswith(varscope.name)]
                self.train_modified.extend(cell_vars)

                predicted_words_for_batch = tf.Variable(np.zeros([0, sequence_len], np.int64), tf.int64,
                                                        name="predicted_words_for_batch")
                empty_word_indexes = tf.Variable(np.zeros([0], np.int64), tf.int64, name="empty_word_indexes")

                logger.debug("LSTM input state bias : %s" % str(biasTensor.shape))

                biasTensor = print_tensor(biasTensor, "total bias tensor", logging.DEBUG)

                for image_index in range(dataset.batch_size_train):
                    if image_index > 0:
                        tf.get_variable_scope().reuse_variables()

                    bias_vector = tf.expand_dims(biasTensor[image_index,:],0)
                    # get the BOS embedding
                    bos_vector = tf.expand_dims(tf.constant(dataset.embedding_matrix[dataset.vocabulary.index('BOS'),:],tf.float32),0)

                    # zero state vector for the initial state

                    current_state = tf.contrib.rnn.LSTMStateTuple(bias_vector, bias_vector)
                    output, current_state = cell(bos_vector, current_state,scope="rnn/basic_lstm_cell")

                    word_embedding, word_index = self.logits_to_word_vectors_tf(dataset.embedding_matrix, logger, fc_out_w,
                                                                                fc_out_b, output,
                                                                                defs.caption_search.max)
                    word_embedding = print_tensor(word_embedding, "word_embedding 0", settings.logging_level)
                    # save predicted word index
                    current_word_indexes = tf.concat([empty_word_indexes, word_index], 0)

                    for step in range(1, sequence_len):

                        tf.get_variable_scope().reuse_variables()

                        logits, current_state = cell(word_embedding, current_state, scope="rnn/basic_lstm_cell")
                        # logger.debug("LSTM iteration #%d state : %s" % (step, [str(x.shape) for x in state]))
                        word_embedding, word_index = self.logits_to_word_vectors_tf(dataset.embedding_matrix, logger,
                                                                                    fc_out_w, fc_out_b, logits,
                                                                                    defs.caption_search.max)
                        # logger.debug("LSTM iteration #%d output : %s" % (step, data_io.shape))
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
            logger = settings.logger
            sequence_len = dataset.max_caption_length + 1  # plus one for the BOS
            dropout_keep_prob = settings.dropout_keep_prob

            dataset.logger.info("Bias tensor : %s" % str(biasTensor.shape))
            biasTensor = print_tensor(biasTensor,"bias tensor", settings.logging_level)

            # need to map the image to the state dimension. If not equal, add an fc layer transformation
            bias_dimension = int(biasTensor.shape[1])
            if bias_dimension != num_hidden:
                logger.info("Mapping visual bias %d-layer to the %d-sized LSTM state." % (bias_dimension, num_hidden))
                # layer initializations
                fc_bias_state_w_init = tf.truncated_normal((bias_dimension, num_hidden), stddev=0.05, name="fc_bias_state_w_init")
                fc_bias_state_b_init = tf.constant(0.1, shape=(num_hidden,), name="fc_bias_state_b_init")

                # create the layers
                fc_bias_state_w = tf.Variable(fc_bias_state_w_init, name="fc_bias_state_w")
                fc_bias_state_b = tf.Variable(fc_bias_state_b_init, name="fc_bias_state_b")

                biasTensor = tf.nn.xw_plus_b(biasTensor, fc_bias_state_w, fc_bias_state_b, name="fc_out")

            dataset.logger.info("Bias tensor post-mapping : %s" % str(biasTensor.shape))
            biasTensor = print_tensor(biasTensor, "bias tensor post mapping",settings.logging_level)
            # probably has to be done with a while on each visual input ...

            # LSTM basic cell
            with tf.variable_scope("LSTM_id_vs") as varscope:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                cell_vars = [v for v in tf.global_variables()
                             if v.name.startswith(varscope.name)]
                self.train_modified.extend(cell_vars)
                logger.debug("LSTM input : %s" % str(wordsTensor.shape))
                logger.debug("LSTM input state bias : %s" % str(biasTensor.shape))

                # get LSTM rawoutput
                output, _ = self.rnn_dynamic(wordsTensor, cell, sequence_len, num_hidden, logger,
                                             settings.logging_level, num_words_caption, biasTensor)
                output = print_tensor(output, "lstm raw output:", settings.logging_level)
                logger.debug("LSTM raw output : %s" % str(output.shape))

                # reshape to num_batches * sequence_len x num_hidden
                output = tf.reshape(output, [-1, num_hidden])
                logger.debug("LSTM recombined output : %s" % str(output.shape))
                output = print_tensor(output, "lstm recombined output", settings.logging_level)

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
                logger.debug("LSTM final output : %s" % str(self.output.shape))
                self.output = print_tensor(self.output, "lstm fc output", settings.logging_level)

                fc_vars = [f for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, namescope)
                           if f.name.startswith(namescope)]
                self.train_modified.extend(fc_vars)

                # include a dummy variables, used in the validation network to enable loading
                predicted_words_for_batch = tf.Variable(np.zeros([0, sequence_len], np.int64), tf.int64,
                                                        name="predicted_words_for_batch")
                empty_word_indexes = tf.Variable(np.zeros([0], np.int64), tf.int64, name="empty_word_indexes")

    def define_lstm_inputbias_validation(self, biasTensor, captions_per_item, dataset, settings):

        with tf.name_scope("LSTM_id") as namescope:
            # num hidden neurons, the size of the hidden state vector
            num_classes = dataset.num_classes
            num_hidden = settings.lstm_num_hidden
            logger = settings.logger
            sequence_len = dataset.max_caption_length + 1  # plus one for the BOS
            dataset.logger.info("Sequence length %d" % sequence_len)

            # need to map the image to the state dimension. If not equal, add an fc layer transformation
            bias_dimension = int(biasTensor.shape[1])
            if bias_dimension != num_hidden:
                logger.info("Mapping visual bias %d-layer to the %d-sized LSTM state." % (bias_dimension, num_hidden))
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
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
                # layer initializations
                # add a final fc layer to convert from num_hidden to a num_classes output
                fc_out__init = tf.truncated_normal((num_hidden, num_classes), stddev=0.05, name="fc_out_w_init")
                fc_out_b_init = tf.constant(0.1, shape=(num_classes,), name="fc_out_b_init")

                # create the layers
                fc_out_w = tf.Variable(fc_out__init, name="fc_out_w", )
                fc_out_b = tf.Variable(fc_out_b_init, name="fc_out_b")

                logger.debug("LSTM bias : %s" % str(biasTensor.shape))

                # in validation mode, we need to have the word embeddings loaded up
                embedding_matrix = tf.constant(dataset.embedding_matrix, tf.float32)
                logger.debug("Loaded the graph embedding matrix : %s" % embedding_matrix.shape)

                # batch_size =  tf.shape(inputTensor)[0]
                # item_index = tf.Variable(-1, tf.int32)
                # images_words_tensors = tf.split(inputTensor, dataset.batch_size_val, 0)

                predicted_words_for_batch = tf.Variable(np.zeros([0, sequence_len], np.int64), tf.int64,
                                                        name="predicted_words_for_batch")
                empty_word_indexes = tf.Variable(np.zeros([0], np.int64), tf.int64, name="empty_word_indexes")
                # assumes fix-sized batches
                for item_index in range(dataset.batch_size_val):

                    # image_word_vector = print_tensor(image_word_vector ,"image_word_vector ",settings.logging_level)
                    # image_word_vector = inputTensor[item_index,:]
                    # image_word_vector = print_tensor(image_word_vector ,"image_word_vector ",settings.logging_level)

                    # item_index = tf.add(item_index , 1)
                    item_index = print_tensor(item_index, "item_index ", settings.logging_level)

                    # image_vector = image_word_vector[0,:image_vector_dim]
                    bias_vector = biasTensor[item_index]
                    # image_vector = tf.slice(inputTensor, [item_index, 0], [1, image_vector_dim])
                    # image_vector = print_tensor(image_vector, "image_vector ", settings.logging_level)
                    # image_vector = print_tensor(image_vector ,"image_vector ",settings.logging_level)

                    # zero state vector for the initial state
                    current_state = tf.contrib.rnn.LSTMStateTuple(bias_vector , bias_vector )
                    output, current_state = cell(inputTensor[item_index:item_index + 1, :], current_state,
                                                 scope="rnn/basic_lstm_cell")

                    word_embedding, word_index = self.logits_to_word_vectors_tf(embedding_matrix, logger,
                                                                                fc_out_w,
                                                                                fc_out_b, output,
                                                                                defs.caption_search.max)
                    word_embedding = print_tensor(word_embedding, "word_embedding 0", settings.logging_level)
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
                        # logger.debug("LSTM iteration #%d state : %s" % (step, [str(x.shape) for x in state]))
                        word_embedding, word_index = self.logits_to_word_vectors_tf(embedding_matrix, logger,
                                                                                    fc_out_w, fc_out_b, logits,
                                                                                    defs.caption_search.max)
                        # logger.debug("LSTM iteration #%d output : %s" % (step, data_io.shape))
                        current_word_indexes = tf.concat([current_word_indexes, word_index], axis=0)
                        word_embedding = print_tensor(word_embedding, "word_embedding %d" % step,
                                                      settings.logging_level)

                    current_word_indexes = print_tensor(current_word_indexes, "current_word_indexes",
                                                        settings.logging_level)
                    # done for the current image, append to batch results
                    predicted_words_for_batch = tf.concat(
                        [predicted_words_for_batch, tf.expand_dims(current_word_indexes, 0)], 0)
                self.output = predicted_words_for_batch


    ## dynamic rnn case, where input is a single tensor
    def rnn_dynamic(self,inputTensor, cell, sequence_len, num_hidden, logger, logging_level, elements_per_sequence = None, init_state=None):
        # data vector dimension
        input_dim = int(inputTensor.shape[-1])

        # reshape input tensor from shape [ num_videos * num_frames_per_vid , input_dim ] to
        # [ num_videos , num_frames_per_vid , input_dim ]
        inputTensor = print_tensor(inputTensor ,"inputTensor  in rnn_dynamic",logging.DEBUG)
        inputTensor = tf.reshape(inputTensor, (-1, sequence_len, input_dim), name="lstm_input_reshape")
        inputTensor = print_tensor(inputTensor, "input reshaped",logging_level)
        logger.info("reshaped inputTensor %s" % str(inputTensor.shape))

        # get the batch size during run. Make zero state to 2 - tuple of [batch_size, num_hidden]
        # 2-tuple state due to the sate_is_tuple LSTM cell
        batch_size = tf.shape(inputTensor)[0]
        if init_state is None:
            zero_state = tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, num_hidden]),tf.zeros([batch_size, num_hidden]))
        else:
            if len(init_state.shape) == 1:
                init_state = tf.expand_dims(init_state,0)
            zero_state = tf.contrib.rnn.LSTMStateTuple(init_state, init_state)

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




