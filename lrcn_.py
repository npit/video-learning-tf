import tensorflow as tf
import os
# models
from models.alexnet import alexnet
from models.lstm import lstm
# util
from utils_ import *


class LRCN:
    # placeholders
    inputData = None
    inputLabels = None
    words_per_image = None
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
        if settings.workflow == defs.workflows.imgdesc:
            self.item_logits = []
            self.item_labels = []
            self.binary_word_idx = tf.placeholder(tf.int32, (None))
        else:
            # initializations
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
            self.logger.error("Weights file %s does not exist." % self.dcnn_weights_file);
            exit("File not found");

        # create the workflow
        if self.workflow == defs.workflows.singleframe:
           self.create_singleframe(settings,dataset)
        elif self.workflow == defs.workflows.lstm:
            self.create_lstm(settings,dataset)
        elif self.workflow == defs.workflows.imgdesc:
            self.create_imgdesc(settings,dataset)
        elif self.workflow == defs.workflows.videodesc:
            self.create_videodesc(settings,dataset)
        else:
            error("Unknown run mode [%s]" % self.workflow)

        # create the training ops
        if settings.do_training:
            self.create_training(settings,summaries)
        self.logger.debug("Completed network definitions.")

    def get_current_lr(self, base_lr, global_step, decay_params):
        if decay_params is None:
            return base_lr
        if decay_params[0] == defs.decay.exp:
            staircase = False
        elif decay_params == defs.decay.staircase:
            staircase = True
        return tf.train.exponential_decay(base_lr, global_step, decay_params[1], decay_params[2],staircase,"lr_decay")

    # training ops
    def create_training(self, settings, summaries):
        # configure loss
        self.logits = print_tensor(self.logits, "training: logits : ")
        # self.inputLabels = print_tensor(self.inputLabels, "training: labels : ")
        with tf.name_scope("cross_entropy_loss"):
            loss_per_vid = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputLabels, name="loss")
            with tf.name_scope('total'):
                self.loss = tf.reduce_mean(loss_per_vid)

            summaries.train.append(add_descriptive_summary(self.loss))

        # configure learning rate
        base_lr = tf.constant(settings.base_lr, tf.float32)
        decay_params = settings.lr_decay
        global_step = tf.Variable(0, tf.int32)
        if settings.lr_decay is not None:
            self.current_lr = self.get_current_lr(settings.base_lr, global_step, decay_params)
        else:
            self.current_lr = base_lr

        # lr per-layer variation
        if settings.lr_mult is not None:
            with tf.name_scope("two_tier_optimizer"):
                # lr decay


                # split tensors to slow and fast learning
                regular_vars, modified_vars  = [], []
                if self.dcnn_model  is not None:
                    regular_vars.extend(self.dcnn_model.train_regular)
                    modified_vars.extend(self.dcnn_model.train_modified)
                if self.lstm_model is not None:
                    regular_vars.extend(self.lstm_model.train_regular)
                    modified_vars.extend(self.lstm_model.train_modified)
                self.logger.info("Setting up two-tier lr training, with modified layers: %s" % str(modified_vars))
                # setup the two optimizers
                if settings.optimizer == defs.optim.sgd:

                    trainer_base = tf.train.GradientDescentOptimizer(self.current_lr,name="sgd_base")\
                        .minimize(self.loss,var_list=regular_vars, global_step=global_step)
                    modified_lr = self.current_lr * settings.lr_mult
                    trainer_modified = tf.train.GradientDescentOptimizer(modified_lr,name="sgd_mod")\
                        .minimize(self.loss,var_list=modified_vars, global_step=global_step)
                else:
                    error("Undefined optimizer %s" % settings.optimizer)

                self.optimizer  = tf.group(trainer_base, trainer_modified)
        else:
            # single lr for all
            self.logger.info("Setting up training with a global learning rate.")
            if settings.optimizer == defs.optim.sgd:
                with tf.name_scope("optimizer"):
                    self.optimizer = tf.train.GradientDescentOptimizer(self.current_lr).minimize(self.loss, global_step=global_step)

        # accuracies
        with tf.name_scope('training_accuracy'):
            with tf.name_scope('correct_prediction_train'):
                # ok for this argmax we gotta squash the labels down to video level.
                correct_predictionTrain = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.inputLabels, 1))
            with tf.name_scope('accuracy_train'):
                self.accuracyTrain = tf.reduce_mean(tf.cast(correct_predictionTrain, tf.float32))

        summaries.train.append(tf.summary.scalar('accuracyTrain', self.accuracyTrain))



    # workflows
    def create_singleframe(self, settings, dataset):
        # define label inputs
        batchLabelsShape = [None, dataset.num_classes]
        self.inputLabels = tf.placeholder(tf.int32, batchLabelsShape, name="input_labels")

        # create the singleframe workflow
        with tf.name_scope("dcnn_workflow"):
            self.logger.info("Dcnn workflow")
            # single DCNN, classifying individual frames
            self.dcnn_model = alexnet.dcnn()
            self.dcnn_model.create(dataset.image_shape, self.dcnn_weights_file, dataset.num_classes)
            self.inputData, framesLogits = self.dcnn_model.get_io()
            self.logger.debug("input : [%s]" % self.inputData.shape)
            self.logger.debug("label : [%s]" % self.inputData.shape)
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
                    self.logger.error("Undefined pooling method: %d " % settings.frame_pooling_type)
                    error("Undefined frame pooling method.")
        else:
            self.logits = framesLogits
            self.logger.info("logits out : [%s]" % self.logits.shape)

    def create_lstm(self,settings, dataset):
        # define label inputs
        batchLabelsShape = [None, dataset.num_classes]
        self.inputLabels = tf.placeholder(tf.int32, batchLabelsShape, name="input_labels")
        # create the lstm workflow
        with tf.name_scope("lstm_workflow"):
            if dataset.input_mode != defs.input_mode.video:
                error("LSTM workflow only available for video input mode")
            # DCNN for frame encoding
            self.dcnn_model = alexnet.dcnn()
            self.dcnn_model.create(dataset.image_shape, self.dcnn_weights_file, dataset.num_classes, settings.lstm_input_layer)
            self.inputData, encodedFrames = self.dcnn_model.get_io()

            self.logger.debug("input : [%s]" % self.inputData.shape)
            self.logger.debug("label : [%s]" % self.inputLabels.shape)
            self.logger.info("dcnn out : [%s]" % encodedFrames.shape)

            # LSTM for frame sequence classification for frame encoding
            self.lstm_model = lstm.lstm()
            self.lstm_model.define_activity_recognition(encodedFrames, dataset, settings)
            self.logits = self.lstm_model.get_output()
            self.logger.info("logits : [%s]" % self.logits.shape)

    def create_imgdesc(self, settings, dataset):

        # implementation below separates train and validation. gotta fix
        # 1 - architectural alternatives are karpathy's image vector h_0 bias
        # 2 - spatial attention backwards lookup (show attend and tell. xu et al. 2015)

        # for two types of evaluating an lstm cell
        # https://stackoverflow.com/questions/37252977/whats-the-difference-between-two-implementations-of-rnn-in-tensorflow

        self.logger.warning("ADD Grad clipping")

        # use the pretrained embedding like this
        # https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow?rq=1
        # create the image description workflow
        with tf.name_scope("imgdesc_workflow"):
            # make sure input mode is image
            if not dataset.input_mode == defs.input_mode.image:
                self.logger.error("The image description workflow works only in image input mode.")
                error("Invalid input mode for image description")
            # set up placeholders

            self.words_per_image = tf.placeholder(tf.int32,shape=(None), name="words_per_image")
            # self.words_per_image = print_tensor(self.words_per_image, "words per image tensor:")

            self.inputLabels = tf.placeholder(tf.int32, [None, dataset.num_classes ], name="input_labels")
            labels = tf.identity(self.inputLabels)
            labels = print_tensor(labels,"input labels")
            self.word_embeddings = tf.placeholder(tf.float32,shape=(None,dataset.embedding_matrix.shape[1]), name="word_embeddings")

            self.logger.debug("input labels : [%s]" % labels)

            # DCNN for frame encoding
            self.dcnn_model = alexnet.dcnn()
            self.dcnn_model.create(dataset.image_shape, self.dcnn_weights_file, dataset.num_classes,settings.lstm_input_layer)
            self.inputData, encodedFrames = self.dcnn_model.get_io()
            self.logger.debug("dcnn input : [%s]" % self.inputData.shape)
            self.logger.debug("dcnn output encodings: [%s]" % encodedFrames.shape)
            encodedFrames = print_tensor(encodedFrames, "encoded frames")

            # if we feed the image into the state, we have to reshape the encoded vector to the state dim
            # this follows the neuraltalk2 architecture

            # if we feed the image into the lstm input, just concat the word embeddings to the images

            frame_encoding_dim = int(encodedFrames.shape[-1])

            if settings.do_training:
                # duplicate the image to the max number of the words in the caption plus 1 for the BOS: concat horizontally
                encodedFrames = tf.tile(encodedFrames, [1, dataset.num_frames_per_clip + 1])
                encodedFrames = print_tensor(encodedFrames, "hor. concatenated frames")
                self.logger.debug("hor. concatenated frames : [%s]" % encodedFrames.shape)
                encodedFrames = tf.reshape(encodedFrames, [-1, frame_encoding_dim], name="restore_to_sequence")
                encodedFrames = print_tensor(encodedFrames, "restored frames")
                self.logger.debug("restored : [%s]" % encodedFrames.shape)


            # horizontal concat the images to the words
            frames_words = tf.concat([encodedFrames, self.word_embeddings],axis=1)
            self.logger.debug("frames concat words : [%s]" % frames_words.shape)
            frames_words = print_tensor(frames_words,"frames concat words ")

            # feed to lstm
            self.lstm_model = lstm.lstm()
            if settings.do_training:
                self.lstm_model.define_image_description(frames_words, frame_encoding_dim, dataset.num_classes, self.words_per_image, dataset, settings)
            else:
                self.lstm_model.define_image_description_validation(frames_words, frame_encoding_dim,
                                                                    dataset.num_classes, self.words_per_image,
                                                                    dataset, settings)
            self.logits = self.lstm_model.get_output()

            # remove the tensor rows where no ground truth caption is present
            self.logger.debug("logits : [%s]" % self.logits.shape)
            self.logits = print_tensor(self.logits ,"logits to process")
            # split the logits to the chunks in the words_per_image. First append the number of rows left to complete
            # the sequence length, so as to subsequently tf.split the tensor

            if settings.do_training:
                # get the number of useless logits per batch item
                num_useless = dataset.num_frames_per_clip - self.words_per_image
                num_useless = print_tensor(num_useless, "num useless")
                wpi = tf.identity(self.words_per_image)
                wpi = wpi + 1
                wpi = print_tensor(wpi , "wpi plus EOS logits postproc")
                # split the logits in [p1,u1,p2,u2,...], where ni,ui the number of predictions and useless output per item
                chunks_sizes = tf.stack([wpi, num_useless])
                chunks_sizes = print_tensor(chunks_sizes , "chunks sizes concatted")

                chunks_sizes = tf.transpose(chunks_sizes)
                chunks_sizes = print_tensor(chunks_sizes , "chunks sizes transposed")

                chunks_sizes = tf.reshape(chunks_sizes,shape=[-1])
                chunks_sizes = tf.cast(chunks_sizes, tf.int32)
                chunks_sizes = print_tensor(chunks_sizes , "chunks sizes ")



                bwi = tf.identity(self.binary_word_idx)
                bwi = print_tensor(bwi,"binary word index to keep")
                self.logits= tf.gather(self.logits, bwi)

                # static_batch_size = dataset.get_batch_size()
                # logit_chunks = tf.split(self.logits, chunks_sizes, num = static_batch_size * 2,  axis=0, name="caption_split")
                # logit_chunks = print_tensor(logit_chunks, "logits list")
                # drop the useless ones; this pre-supposes the obvious that each item has at least one caption word
                self.logits = print_tensor(self.logits, "filtered logits list")
                # re-merge the tensor list into a tensor
                self.logits = tf.concat(self.logits, axis=0)
                self.logits = print_tensor(self.logits,"final filtered logits")
                self.logger.debug("final filtered logits : [%s]" % self.logits.shape)
            else:
                # for validation, just return the logits and we'll process them on the driver
                pass

    def create_videodesc(self, settings, dataset):
        pass
    # validation accuracy computation
    def process_validation_logits(self, logits, dataset, labels):
        # processing for image description
        if self.workflow == defs.workflows.imgdesc:
            eos_index = dataset.vocabulary.index("EOS")
            # logits is words
            for i in range(0,len(logits), dataset.num_frames_per_clip):
                logits_seq = logits[i:i+dataset.num_frames_per_clip]
                # one is 1, at the position of the eos
                eos_position = [1 if x == eos_index else 0 for x in logits_seq]

                if any(eos_position):
                    # keep up to but not including eos. Get first index, if multiple .
                    eos_index = eos_position.index(1)
                    logits_seq = logits_seq[0:eos_index]
                # else, no EOS exists in the predicted caption
                # append the vector
                self.item_logits.append(logits_seq)
                self.item_labels.append(labels)
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
                    self.logger.error("Logits and/or labels non empty at the end of video item mode aggregation!")
                    error("Video item mode aggregation error.")
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
