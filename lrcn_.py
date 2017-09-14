import tensorflow as tf
import  math
# models
from models.alexnet import alexnet
from models.lstm import lstm
# util
from utils_ import *


class LRCN:
    # placeholders
    inputData = None
    inputLabels = None
    caption_lengths = None
    workflow = None

    # output data
    logits = None
    accuracy = None

    # internal
    loss = None
    optimizer = None

    logger = None
    item_logits = None
    item_labels = None
    clip_logits = None
    clip_labels = None
    clip_pooling_type = None

    current_lr = None
    dcnn_model = None
    lstm_model = None

    dcnn_weights_file = None
    # let there be network
    def create(self, settings, dataset, summaries):
        # initializations
        if defs.workflows.is_description(settings.workflow):
            self.item_logits = []
            self.item_labels = []
            self.non_padding_word_idxs = tf.placeholder(tf.int32, (None))
        else:
            # items refer to the primary unit we operate one, i.e. videos or frames
            self.item_logits = np.zeros([0, dataset.num_classes], np.float32)
            self.item_labels = np.zeros([0, dataset.num_classes], np.float32)
            # clips refers to image groups that compose a video, for training with clip information
            self.clip_logits = np.zeros([0, dataset.num_classes], np.float32)
            self.clip_labels = np.zeros([0, dataset.num_classes], np.float32)
            self.clip_pooling_type = settings.clip_pooling_type

        # define network input
        self.logger = settings.logger
        self.workflow = settings.workflow

        # make sure dcnn weights are good2go
        self.dcnn_weights_file = os.path.join(os.getcwd(), "models/alexnet/bvlc_alexnet.npy")
        if not os.path.exists(self.dcnn_weights_file):
            error("Weights file %s does not exist." % self.dcnn_weights_file, self.logger)

        # create the workflow
        if self.workflow == defs.workflows.acrec.singleframe:
           self.create_actrec_singleframe(settings, dataset)
        elif self.workflow == defs.workflows.acrec.lstm:
            self.create_actrec_lstm(settings, dataset)

        elif self.workflow == defs.workflows.imgdesc.inputstep:
            self.create_imgdesc_visualinput(settings, dataset)
        elif self.workflow == defs.workflows.imgdesc.statebias:
            self.create_imgdesc_statebias(settings, dataset)

        elif self.workflow == defs.workflows.videodesc.pooled:
            self.create_videodesc_pooling(settings,dataset)
        elif self.workflow == defs.workflows.videodesc.encdec:
            self.create_videodesc_encdec(settings,dataset)

        else:
            error("Unknown run mode [%s]" % self.workflow)

        # create the training ops
        if settings.do_training:
            self.create_training(settings, dataset, summaries)

    def precompute_learning_rates(self, settings, dataset):
        self.logger.info("Precomputing learning rates per batch")

        base_lr = settings.base_lr
        decay_params = settings.lr_decay
        num_batches = len(dataset.batches_train)
        total_num_batches = num_batches * dataset.epochs
        lr_per_batch = []
        if decay_params is None:
            return [base_lr for _ in range(total_num_batches)]
        log_message = "Dropping LR "
        lr_drop_offset = 0 if len(tuple(decay_params)) == 4 else decay_params[-1]
        decay_strategy, decay_scheme, decay_freq, decay_factor = tuple(decay_params[:4])

        if decay_strategy == defs.decay.granularity.exp:
            staircase = False
            log_message += "smoothly "
        elif decay_strategy == defs.decay.granularity.staircase:
            staircase = True
            log_message += "jaggedly "
        else:
            error("Undefined decay strategy %s" % decay_strategy, settings.logger)

        if decay_scheme == defs.decay.scheme.interval:
            # reduce every decay_freq batches
            decay_period = decay_freq
            log_message += "every %d step(s) " % decay_period
        elif decay_scheme == defs.decay.scheme.total:
            # reduce a total of decay_freq times
            decay_period = math.ceil(total_num_batches / decay_freq)
            log_message += "every ceil[(%d batches x %d epochs) / %d total steps] = %d steps" % \
                (num_batches, dataset.epochs, decay_freq, decay_period)
        else:
            error("Undefined decay scheme %s" % decay_scheme, settings.logger)

        idx = 0
        while len(lr_per_batch) < total_num_batches:
            if staircase:
                fraction = idx // decay_freq
            else:
                fraction = idx / decay_freq
            current_lr = base_lr * pow(decay_factor,fraction)
            idx = idx + decay_freq
            lr_per_batch.extend([current_lr for _ in range(decay_period)])

        lr_per_batch = lr_per_batch[:total_num_batches]
        if lr_drop_offset:
            lr_per_batch = [base_lr for _ in range(lr_drop_offset)] + lr_per_batch[0:-lr_drop_offset]
            log_message += " - with a %d-step offset " % lr_drop_offset

        lr_schedule_file = os.path.join(settings.run_folder,settings.run_id + "_lr_decay_schedule.txt")
        with open(lr_schedule_file,"w") as f:
            batches = [ x for _ in range(dataset.epochs) for x in range(num_batches)]
            if len(batches) != total_num_batches:
                error("Batch length precomputation mismatch",  settings.logger)
            epochs = [ep for ep in range(dataset.epochs) for _ in dataset.batches_train]
            batches_lr = list(zip(epochs, batches, lr_per_batch))

            for b in batches_lr:
                f.write("Epoch %d/%d, batch %d/%d, lr %2.8f\n" % (b[0]+1,dataset.epochs,b[1]+1, num_batches,b[2]))
        self.logger.info(log_message)
        return lr_per_batch


    # training ops
    def create_training(self, settings, dataset, summaries):

        self.logits = print_tensor(self.logits, "training: logits : ", settings.logging_level)
        # self.inputLabels = print_tensor(self.inputLabels, "training: labels : ",settings.logging_level)

        # configure loss
        with tf.name_scope("cross_entropy_loss"):
            loss_per_vid = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputLabels, name="loss")
            with tf.name_scope('total'):
                self.loss = tf.reduce_mean(loss_per_vid)
            summaries.train.append(add_descriptive_summary(self.loss))

        # configure the learning rate
        learning_rates = self.precompute_learning_rates(settings,dataset)
        self.learning_rates = tf.constant(learning_rates,tf.float32,name="Learning_rates")
        self.global_step = tf.Variable(0, dtype = tf.int32, trainable=False,name="global_step")
        with tf.name_scope("lr"):
            self.current_lr = self.learning_rates[self.global_step ]
            summaries.train.append(add_descriptive_summary(self.current_lr))

        # setup the training ops, with a potential lr per-layer variation
        if settings.lr_mult is not None:
            self.create_multi_tier_learning(settings, dataset, summaries)
        else:
            self.create_single_tier_learning(settings, dataset, summaries)

        # accuracies
        with tf.name_scope('training_accuracy'):
            with tf.name_scope('correct_prediction_train'):
                # ok for this argmax we gotta squash the labels down to video level.
                correct_predictionTrain = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.inputLabels, 1))
            with tf.name_scope('accuracy_train'):
                self.accuracyTrain = tf.reduce_mean(tf.cast(correct_predictionTrain, tf.float32))

        summaries.train.append(tf.summary.scalar('accuracyTrain', self.accuracyTrain))

    # establish the training ops with slow and fast learning parameters
    def create_multi_tier_learning(self, settings, dataset, summaries):
        with tf.name_scope("two_tier_optimizer"):
            # split tensors to slow and fast learning, as per their definition
            regular_vars, modified_vars = [], []
            if self.dcnn_model is not None:
                regular_vars.extend(self.dcnn_model.train_regular)
                modified_vars.extend(self.dcnn_model.train_modified)
            if self.lstm_model is not None:
                regular_vars.extend(self.lstm_model.train_regular)
                modified_vars.extend(self.lstm_model.train_modified)
            self.logger.info("Setting up two-tier training with a factor of %f for the %d layer(s): %s" % (
            settings.lr_mult, len(modified_vars), [ m.name for m in modified_vars]))

            # setup the two optimizer
            if settings.optimizer == defs.optim.sgd:
                opt = tf.train.GradientDescentOptimizer(self.current_lr, name="sgd_base")
            elif settings.optimizer == defs.optim.adam:
                opt = tf.train.AdamOptimizer(self.current_lr)
            else:
                error("Undefined optimizer %s" % settings.optimizer)

            # computer two-tier grads
            grads = opt.compute_gradients(self.loss, var_list=regular_vars)
            if settings.clip_grads is not None:
                clipmin, clipmax = settings.clip_grads
                grads = [(tf.clip_by_norm(grad, clipmax), var) for grad, var in grads]
            trainer_base = opt.apply_gradients(grads)

            modified_lr = self.current_lr * settings.lr_mult
            opt_mod = tf.train.GradientDescentOptimizer(modified_lr, name="sgd_mod")
            grads_mod = opt_mod.compute_gradients(self.loss, var_list=modified_vars)
            if settings.clip_grads is not None:
                clipmin, clipmax = settings.clip_grads
                grads_mod = [(tf.clip_by_norm(grad_mod, clipmax), var_mod) for grad_mod, var_mod in
                             grads_mod]
            trainer_modified = opt.apply_gradients(grads_mod, global_step=self.global_step)



            self.optimizer = tf.group(trainer_base, trainer_modified)
        with tf.name_scope("grads_norm"):
            grads_norm = tf.reduce_mean(list(map(tf.norm, grads)))
            summaries.train.append(add_descriptive_summary(grads_norm))
        with tf.name_scope("grads_mod_norm"):
            grads_mod_norm = tf.reduce_mean(list(map(tf.norm, grads_mod)))
            summaries.train.append(add_descriptive_summary(grads_mod_norm))

    def create_single_tier_learning(self, settings, dataset, summaries):
        # single lr for all
        self.logger.info("Setting up training with a global learning rate.")
        with tf.name_scope("single_tier_optimizer"):
            if settings.optimizer == defs.optim.sgd:
                opt = tf.train.GradientDescentOptimizer(self.current_lr)
            elif settings.optimizer == defs.optim.adam:
                opt = tf.train.AdamOptimizer(self.current_lr)
            else:
                error("Undefined optimizer %s" % settings.optimizer)

            grads = opt.compute_gradients(self.loss)
            if settings.clip_grads is not None:
                clipmin, clipmax = settings.clip_grads
                grads = [(tf.clip_by_value(grad, clipmin, clipmax), var) for grad, var in grads]
            self.optimizer = opt.apply_gradients(grads, global_step=self.global_step)

        with tf.name_scope('grads_norm'):
            grads_norm = tf.reduce_mean(list(map(tf.norm, grads)))
            summaries.train.append(add_descriptive_summary(grads_norm))

    # workflows

    # Activity recognition
    def create_actrec_singleframe(self, settings, dataset):
        # define label inputs
        batchLabelsShape = [None, dataset.num_classes]
        self.inputLabels = tf.placeholder(tf.int32, batchLabelsShape, name="input_labels")

        settings.frame_encoding_layer = None
        # create the singleframe workflow
        with tf.name_scope("dcnn_workflow"):
            self.logger.info("Dcnn workflow")
            # single DCNN, classifying individual frames
            self.inputData, framesLogits = self.make_dcnn(dataset,settings)
            # do video level pooling only if necessary
        if dataset.input_mode == defs.input_mode.video:
            # average the logits on the frames dimension
            with tf.name_scope("video_level_pooling"):
                if settings.frame_pooling_type == defs.pooling.avg:
                    # -1 on the number of videos (batchsize) to deal with varying values for test and train
                    self.logger.info("Raw logits : [%s]" % framesLogits.shape)
                    frames_per_item = dataset.num_frames_per_clip if dataset.input_mode == defs.input_mode.video else 1

                    frameLogits = tf.reshape(framesLogits, (-1, frames_per_item, dataset.num_classes),
                                             name="reshape_framelogits_pervideo")
                    self.logits = tf.reduce_mean(frameLogits, axis=1)

                    self.logger.info("Averaged logits out : [%s]" % self.logits.shape)
                elif settings.frame_pooling_type == defs.pooling.last:
                    # keep only the response at the last time step
                    self.logits = tf.slice(framesLogits, [0, dataset.num_frames_per_clip - 1, 0],
                                           [-1, 1, dataset.num_classes],
                                           name="last_pooling")
                    self.logger.debug("Last-liced logits: %s" % str(self.logits.shape))
                    # squeeze empty dimension to get vector
                    output = tf.squeeze(self.logits, axis=1, name="last_pooling_squeeze")
                    self.logger.debug("Last-sliced squeezed out : %s" % str(output.shape))
                else:
                    error("Undefined pooling method: %d " % settings.frame_pooling_type, self.logger)
        else:
            self.logits = framesLogits
            self.logger.info("logits out : [%s]" % self.logits.shape)

    def create_actrec_lstm(self, settings, dataset):
        # define label inputs
        batchLabelsShape = [None, dataset.num_classes]
        settings.frame_encoding_layer = None
        self.inputLabels = tf.placeholder(tf.int32, batchLabelsShape, name="input_labels")
        # create the lstm workflow
        with tf.name_scope("lstm_workflow"):
            if dataset.input_mode != defs.input_mode.video:
                error("LSTM workflow only available for video input mode", self.logger)

            # DCNN for frame encoding
            self.inputData, encodedFrames = self.make_dcnn(settings, dataset)


            # LSTM for frame sequence classification for frame encoding
            self.lstm_model = lstm.lstm()
            self.lstm_model.define_activity_recognition(encodedFrames, dataset, settings)
            self.logits = self.lstm_model.get_output()
            self.logger.info("logits : [%s]" % self.logits.shape)

    # Image description
    def create_imgdesc_visualinput(self, settings, dataset):

        with tf.name_scope("imgdesc_workflow"):
            # make sure input mode is image
            if dataset.input_mode != defs.input_mode.image:
                error("The image description workflow works only in image input mode.", self.logger)

            self.make_imgdesc_placeholders(settings,dataset)

            self.inputData, encodedFrames = self.make_dcnn(dataset,settings)
            self.make_imgdesc_early_fusion(settings,dataset,encodedFrames)

    def create_imgdesc_statebias(self, settings, dataset):
        # the implementation here implements the "show and tell model"
        # make sure input mode is image
        if dataset.input_mode != defs.input_mode.image:
            error("The image description workflow works only in image input mode.", self.logger)
        with tf.name_scope("imgdesc_workflow"):
            self.make_imgdesc_placeholders(settings,dataset)
            self.inputData, encodedFrames = self.make_dcnn(dataset,settings)
            self.make_imgdesc_statebias(settings, dataset, encodedFrames)

    def make_imgdesc_placeholders(self, settings, dataset):
        # set up placeholders
        self.caption_lengths = tf.placeholder(tf.int32, shape=(None), name="words_per_item")
        self.inputLabels = tf.placeholder(tf.int32, [None, dataset.num_classes], name="input_labels")
        labels = tf.identity(self.inputLabels)
        labels = print_tensor(labels, "input labels", settings.logging_level)
        self.word_embeddings = tf.placeholder(tf.float32, shape=(None, dataset.embedding_matrix.shape[1]),
                                              name="word_embeddings")
        self.logger.debug("input labels : [%s]" % labels)

    def make_imgdesc_statebias(self, settings, dataset, encodedFrames):

        # make lstm
        self.lstm_model = lstm.lstm()
        if settings.do_training:
            # self.lstm_model.define_lstm_inputbias_loop(self.word_embeddings, encodedFrames, self.caption_lengths, dataset, settings)
            self.lstm_model.define_lstm_inputbias(self.word_embeddings, encodedFrames, self.caption_lengths, dataset, settings)
        else:
            self.lstm_model.define_lstm_inputbias_loop_validation( encodedFrames, self.caption_lengths, dataset, settings)

        self.logits = self.lstm_model.get_output()
        # drop padding logits for training mode
        if settings.do_training:
            self.process_description_training_logits(settings)

    def make_imgdesc_early_fusion(self, settings, dataset, encodedFrames):
        frame_encoding_dim = int(encodedFrames.shape[-1])

        if settings.do_training:
            # duplicate the image to the max number of the words in the caption plus 1 for the BOS: concat horizontally
            encodedFrames = tf.tile(encodedFrames, [1, dataset.max_caption_length + 1])
            encodedFrames = print_tensor(encodedFrames, "hor. concatenated frames", settings.logging_level)
            self.logger.debug("hor. concatenated frames : [%s]" % encodedFrames.shape)
            encodedFrames = tf.reshape(encodedFrames, [-1, frame_encoding_dim], name="restore_to_sequence")
            encodedFrames = print_tensor(encodedFrames, "restored frames", settings.logging_level)
            self.logger.debug("restored : [%s]" % encodedFrames.shape)

        # horizontal concat the images to the words
        frames_words = tf.concat([encodedFrames, self.word_embeddings], axis=1)
        self.logger.debug("frames concat words : [%s]" % frames_words.shape)
        frames_words = print_tensor(frames_words, "frames concat words ", settings.logging_level)

        # feed to lstm
        self.lstm_model = lstm.lstm()
        if settings.do_training:
            self.lstm_model.define_imgdesc_inputstep(frames_words, frame_encoding_dim,
                                                     self.caption_lengths, dataset, settings)
        else:
            self.lstm_model.define_imgdesc_inputstep_validation(frames_words, frame_encoding_dim,
                                                                self.caption_lengths,
                                                                dataset, settings)

        self.logits = self.lstm_model.get_output()

        # remove the tensor rows where no ground truth caption is present
        self.logger.debug("logits : [%s]" % self.logits.shape)
        self.logits = print_tensor(self.logits, "logits to process", settings.logging_level)
        # split the logits to the chunks in the caption_lengths. First append the number of rows left to complete
        # the sequence length, so as to subsequently tf.split the tensor

        if settings.do_training:
            self.process_description_training_logits(settings)

    def process_description_training_logits(self, settings):
        # remove the logits corresponding to padding
        non_padding_logits = tf.identity(self.non_padding_word_idxs)
        non_padding_logits = print_tensor(non_padding_logits, "non-padding word idxs to keep", settings.logging_level)
        self.logits = tf.gather(self.logits, non_padding_logits)
        self.logits = print_tensor(self.logits, "filtered logits list", settings.logging_level)
        # re-merge the tensor list into a tensor
        self.logits = tf.concat(self.logits, axis=0)
        self.logits = print_tensor(self.logits, "final filtered logits", settings.logging_level)
        self.logger.debug("final filtered logits : [%s]" % self.logits.shape)

    def make_dcnn(self, dataset, settings):
        # DCNN for frame encoding
        self.dcnn_model = alexnet.dcnn()
        self.dcnn_model.create(dataset.image_shape, self.dcnn_weights_file, dataset.num_classes,
                               settings.frame_encoding_layer)
        inputData, outputData = self.dcnn_model.get_io()
        self.logger.debug("dcnn input : [%s]" % inputData.shape)
        self.logger.debug("dcnn output : [%s]" % outputData.shape)
        outputData = print_tensor(outputData, "encoded frames", settings.logging_level)
        return inputData, outputData

    # video description
    def create_videodesc_pooling(self, settings, dataset):
        # Venugopalan et al. 2015 : pool video frames, do img desc.

        # get dcnn encodings for each frame
        self.inputData, encoded_frames = self.make_dcnn(dataset, settings)

        # pool video frames to a single vector
        if settings.frame_pooling_type == defs.pooling.avg:
            pooled_frame = tf.reduce_mean(encoded_frames, 0)
        else:
            error("Undefined pooling type %s" % settings.frame_pooling_type, self.logger)
        # the rest of the workflow is identical to the image description statebias workflow
        self.make_imgdesc_placeholders(settings, dataset)
        self.make_imgdesc_prepro(settings,dataset,pooled_frame)

    def create_videodesc_encdec(self, settings, dataset):
        # Venugopalan et al. 2016 : lstm encoder - decoder

        # get dcnn encodings for each frame
        self.inputData, encoded_frames = self.make_dcnn(dataset, settings)
        # frame sequence is mapped to a fixed-length vector via an lstm
        encoder = lstm.lstm()
        encoder.define_encoder(encoded_frames, settings, dataset)
        encoded_state = encoder.get_output()
        self.make_imgdesc_placeholders(settings, dataset)
        # the rest of the workflow is identical to the image description workflow
        self.make_imgdesc_prepro(settings, dataset, encoded_state)




    # validation accuracy computation
    def process_validation_logits(self, logits, dataset, fdict, padding):
        labels = fdict[self.inputLabels]
        # processing for image description
        if defs.workflows.is_description(self.workflow):
            caption_lengths = fdict[self.caption_lengths]
            assert (len(logits) - padding == len(caption_lengths)), "Logits, labels length mismatch (%d, %d)" % (len(logits)-padding, len(caption_lengths))
            eos_index = dataset.vocabulary.index("EOS")
            # logits is words
            for idx in range(len(logits) - padding):
                image_logits = logits[idx,:]
                image_logits = image_logits[:dataset.max_caption_length]
                eos_position_binary_idx = [1 if x == eos_index else 0 for x in image_logits]

                if any(eos_position_binary_idx):
                    # keep up to but not including eos. Get first index, if multiple .
                    first_eos = eos_position_binary_idx.index(1)
                    image_logits = image_logits[0:first_eos]
                # else, no EOS exists in the predicted caption
                # append the vector
                self.item_logits.append(image_logits)

            # get the labels. In validation mode, labels are EOS-free.
            cumulative_offset = 0
            for item_idx, cap_len in enumerate(caption_lengths):
                label_idxs = [ x + cumulative_offset for x in list(range(cap_len))]
                item_labels = labels[label_idxs,:]
                self.item_labels.append(item_labels)
                cumulative_offset = cumulative_offset + cap_len
            return

        # batch item contains logits that correspond to whole clips. Accumulate to clip storage, and check for aggregation.
        if dataset.batch_item == defs.batch_item.clip:
            # per-clip logits in input : append to clip logits accumulator
            self.clip_logits = np.vstack((self.clip_logits, logits))
            self.clip_labels = np.vstack((self.clip_labels, labels))
            self.logger.debug("Adding %d,%d clip logits and labels to a total of %d,%d." % (
                logits.shape[0], labels.shape[0], self.clip_logits.shape[0], self.clip_labels.shape[0]))

            cpv = dataset.clips_per_video[dataset.video_index]
            # while possible, pop a chunk for the current cpv, aggregate, and add to video logits accumulator
            while dataset.video_index < len(dataset.clips_per_video) and cpv <= len(self.clip_logits):

                # aggregate the logits and add to video logits accumulation
                self.apply_clip_pooling(self.clip_logits, cpv, self.clip_labels)
                # delete them from the accumulation
                self.clip_logits = self.clip_logits[cpv:,:]
                self.clip_labels = self.clip_labels[cpv:,:]

                self.logger.debug("Aggregated %d clips to the %d-th video. Video accumulation is now %d,%d - clip accumulation is %d, %d." %
                                  (dataset.clips_per_video[dataset.video_index], 1 + dataset.video_index, len(self.item_logits),
                                   len(self.item_labels), len(self.clip_logits), len(self.clip_labels)))
                # advance video index
                dataset.video_index = dataset.video_index + 1
                if dataset.video_index >= len(dataset.clips_per_video):
                    break
                cpv = dataset.clips_per_video[dataset.video_index]
        else:
            # batch items are whole items of data
            if dataset.input_mode == defs.input_mode.video:
                # can directly pool and append to video accumulators
                maxvid = dataset.batch_index * dataset.batch_size
                minvid = maxvid - dataset.batch_size

                for vidx in range(minvid, maxvid):
                    if vidx >= dataset.get_num_items():
                        break
                    cpv = dataset.clips_per_video[vidx]
                    self.logger.debug("Aggregating %d clips for video %d in video batch mode" % (cpv, vidx + 1))
                    self.apply_clip_pooling(logits,cpv,labels)
                    logits = logits[cpv:,:]
                    labels = labels[cpv:,:]
                if not (len(logits) == 0 and len(labels) == 0):
                    error("Logits and/or labels non empty at the end of video item mode aggregation!", self.logger)
                self.logger.debug("Video logits and labels accumulation is now %d,%d video in video batch mode." %
                                  (len(self.item_logits), len(self.item_labels)))
            else:
                # frames, simply append
                self.add_item_logits_labels(logits,labels)

    def apply_clip_pooling(self, clips_logits, cpv, video_labels):
        curr_clips = clips_logits[0:cpv,:]
        video_label = video_labels[0,:]
        if self.clip_pooling_type == defs.pooling.avg:
            video_logits = np.mean(curr_clips , axis=0)
        elif self.clip_pooling_type == defs.pooling.last:
            video_logits = curr_clips [-1, :]
        # add logits, label to the video accumulation
        self.add_item_logits_labels(video_logits, video_label)

    def add_item_logits_labels(self,logits,label):
        # add logits, label to the video accumulation
        self.item_logits = np.vstack((self.item_logits, logits))
        self.item_labels = np.vstack((self.item_labels, label))

    def get_accuracy(self):
        # compute accuracy
        self.logger.info("Computing accuracy out of %d items" % len(self.item_logits))
        predicted_classes = np.argmax(self.item_logits, axis=1)
        correct_classes = np.argmax(self.item_labels, axis=1)
        accuracy = np.mean(np.equal(predicted_classes, correct_classes))
        return accuracy
