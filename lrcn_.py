import tensorflow as tf
import os
# models
from models.alexnet import alexnet
from models.lstm import lstm
# util
from utils_ import *


class LRCN:
    logitsTrain = None
    logitsTest = None
    inputData = None
    inputLabels = None
    outputTensor = None
    accuracy = None
    loss = None
    optimizer = None
    logits = None
    logger = None
    item_logits = None
    item_labels = None
    clip_logits = None
    clip_labels = None
    clip_pooling_type = None
    # let there be network
    def create(self, settings, dataset, run_mode,summaries):
        self.item_logits = np.zeros([0, dataset.num_classes], np.float32)
        self.item_labels = np.zeros([0, dataset.num_classes], np.float32)
        self.clip_logits = np.zeros([0, dataset.num_classes ], np.float32)
        self.clip_labels = np.zeros([0, dataset.num_classes ], np.float32)
        self.clip_pooling_type = settings.clip_pooling_type

        self.logger = settings.logger
        batchLabelsShape = [None, dataset.num_classes]
        self.inputLabels = tf.placeholder(tf.int32, batchLabelsShape, name="input_labels")
        weightsFile = os.path.join(os.getcwd(), "models/alexnet/bvlc_alexnet.npy")
        if not os.path.exists(weightsFile):
            self.logger.error("Weights file %s does not exist." % weightsFile);
            exit("File not found");

        if run_mode == defs.run_types.singleframe:

            with tf.name_scope("dcnn_workflow"):
                self.logger.info("Dcnn workflow")
                # single DCNN, classifying individual frames
                self.inputData, framesLogits = alexnet.define(dataset.image_shape, weightsFile, dataset.num_classes)
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
                        frameLogits = tf.reshape(framesLogits, (-1, frames_per_item , dataset.num_classes),
                                                 name="reshape_framelogits_pervideo")

                        self.logits = tf.scalar_mul(1 / frames_per_item, tf.reduce_sum(frameLogits, axis=1))
                        self.logger.info("Averaged logits out : [%s]" % self.logits.shape)
                    elif settings.frame_pooling_type == defs.pooling.last:
                        # keep only the response at the last time step
                        self.logits = tf.slice(framesLogits, [0, dataset.num_frames_per_clip - 1, 0], [-1, 1, dataset.num_classes],
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


        elif run_mode == defs.run_types.lstm:
            with tf.name_scope("lstm_workflow"):
                if dataset.input_mode != defs.input_mode.video:
                    error("LSTM workflow only available for video input mode")
                #  DCNN for frame encoding
                self.inputData, self.outputTensor = alexnet.define(dataset.image_shape, weightsFile, dataset.num_classes,settings.lstm_input_layer)
                self.logger.debug("input : [%s]" % self.inputData.shape)
                self.logger.debug("label : [%s]" % self.inputData.shape)
                self.logger.info("dcnn out : [%s]" % self.outputTensor.shape)

                # LSTM for frame sequence classification for frame encoding
                self.logits = lstm.define(self.outputTensor, dataset, settings, summaries)
                self.logger.info("logits : [%s]" % self.logits.shape)


        else:
            error("Unknown run mode [%s]" % run_mode)



        # loss
        with tf.name_scope("cross_entropy_loss"):
            loss_per_vid = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputLabels, name="loss")
            with tf.name_scope('total'):
                self.loss = tf.reduce_mean(loss_per_vid)

            summaries.train.append(add_descriptive_summary(self.loss))

        # optimization
        if settings.optimizer == "SGD":
            with tf.name_scope("optimizer"):
                self.optimizer = tf.train.GradientDescentOptimizer(settings.learning_rate).minimize(self.loss)


        # accuracies
        with tf.name_scope('training_accuracy'):
            with tf.name_scope('correct_prediction_train'):

                #ok for this argmax we gotta squash the labels down to video level.
                correct_predictionTrain = tf.equal(tf.argmax(self.logits , 1), tf.argmax(self.inputLabels, 1))
            with tf.name_scope('accuracy_train'):
                self.accuracyTrain = tf.reduce_mean(tf.cast(correct_predictionTrain, tf.float32))

        # with tf.name_scope('validation_accuracy'):
        #
        #     with tf.name_scope('correct_prediction_val'):
        #
        #         #ok for this argmax we gotta squash the labels down to video level.
        #         correct_predictionVal = tf.equal(tf.argmax(self.logits , 1), tf.argmax(self.inputLabels, 1))
        #     with tf.name_scope('accuracy_val'):
        #         self.accuracyVal = tf.reduce_mean(tf.cast(correct_predictionVal, tf.float32))
        #
        # summaries.val.append(tf.summary.scalar('accuracyVal', self.accuracyVal))
        summaries.train.append(tf.summary.scalar('accuracyTrain', self.accuracyTrain))
        self.logger.debug("Completed network definitions.")

    # accuracy computation for videos with multiple clips, where logits are already fetched
    def process_validation_logits(self, logits, dataset, labels):
        # batch item contains individual clips. Accumulate to clip storage, and check for aggregation
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
        print(self.item_logits)
        self.logger.info("Computing accuracy out of %d items" % len(self.item_logits))
        predicted_classes = np.argmax(self.item_logits, axis=1)
        correct_classes = np.argmax(self.item_labels, axis=1)
        accuracy = np.mean(np.equal(predicted_classes, correct_classes))
        return accuracy
