import tensorflow as tf
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

    # let there be network
    def create(self, settings, dataset, run_mode,summaries):
        self.logger = settings.logger
        batchLabelsShape = [None, dataset.num_classes]
        self.inputLabels = tf.placeholder(tf.int32, batchLabelsShape, name="input_labels")
        weightsFile = "/home/nik/uoa/msc-thesis/implementation/models/alexnet/bvlc_alexnet.npy"


        if run_mode == defs.run_types.singleframe:
            with tf.name_scope("dcnn_workflow"):
                self.logger.info("Dcnn workflow")
                # single DCNN, classifying individual frames
                self.inputData, framesLogits = alexnet.define(dataset.image_shape, weightsFile, dataset.num_classes)
                self.logger.info("input : [%s]" % self.inputData.shape)
                self.logger.info("label : [%s]" % self.inputData.shape)
            # do video level pooling only if necessary
            if dataset.input_mode == defs.input_mode.video:
                # average the logits on the frames dimension
                with tf.name_scope("video_level_pooling"):
                    # -1 on the number of videos (batchsize) to deal with varying values for test and train
                    self.logger.info("raw per-frame logits : [%s]" % framesLogits.shape)
                    frames_per_item = dataset.num_frames_per_video if dataset.input_mode == defs.input_mode.video else 1
                    frameLogits = tf.reshape(framesLogits, (-1, frames_per_item , dataset.num_classes),
                                             name="reshape_framelogits_pervideo")

                    self.logits = tf.scalar_mul(1 / frames_per_item, tf.reduce_sum(frameLogits, axis=1))
                    self.logger.info("logits out : [%s]" % self.logits.shape)
            else:
                self.logits = framesLogits
                self.logger.info("logits out : [%s]" % self.logits.shape)


        elif run_mode == defs.run_types.lstm:
            with tf.name_scope("lstm_workflow"):
                if dataset.input_mode!= defs.input_mode.video:
                    error("LSTM workflow only available for video input mode")
                #  DCNN for frame encoding
                self.inputData, self.outputTensor = alexnet.define(dataset.image_shape, weightsFile, dataset.num_classes,settings.lstm_input_layer)
                self.logger.info("input : [%s]" % self.inputData.shape)
                self.logger.info("label : [%s]" % self.inputData.shape)
                self.logger.info("dcnn out : [%s]" % self.outputTensor.shape)

                # LSTM for frame sequence classification for frame encoding
                self.logits = lstm.define(self.outputTensor, dataset.get_batch_size(), dataset.num_classes)
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




        with tf.name_scope('validation_accuracy'):
            with tf.name_scope('correct_prediction_val'):

                #ok for this argmax we gotta squash the labels down to video level.
                correct_predictionVal = tf.equal(tf.argmax(self.logits , 1), tf.argmax(self.inputLabels, 1))
            with tf.name_scope('accuracy_val'):
                self.accuracyVal = tf.reduce_mean(tf.cast(correct_predictionVal, tf.float32))

        summaries.train.append(tf.summary.scalar('accuracyTrain', self.accuracyTrain))
        summaries.val.append(tf.summary.scalar('accuracyVal', self.accuracyVal))
        print()

