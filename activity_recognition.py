# essential
import tensorflow as tf
import numpy as np
import math
import os
import argparse

# shufflin'
from random import shuffle
# image IO
from scipy.misc import imread, imresize, imsave
# displaying
import matplotlib.pyplot as plt
# generic IO
import pickle

# models
from models.alexnet import alexnet
from models.lstm import lstm

# utils
from utils import *


# global variables
##################
# tensor list to print
print_tensors = []
# summaries for training & validation
train_summaries_list = []
val_summaries_list = []

# settings class
################
# Generic run settings and parameters should go here
#

class Settings:
    run_id = "test"
    networkName = "alexnet"
    epochs = 10
    batchSize = 128
    tensorboard_folder = "/home/nik/uoa/msc-thesis/implementation/tensorboard_graphs"

    verbosity = 1

    # data input format
    dataFormat = "TFRecord" # TFRecord or raw
    dataFormat = "raw"  # TFRecord or raw

    # run modes
    BASELINE = "baseline"
    LSTM = "lstm"
    lstm_input_layer = "fc7"

    # optimization method and params
    optimizer = "SGD"
    learning_rate = 0.001

    phase = None
    # test interval
    testEvery = 50
    saver = tf.train.Saver()
    do_resume = False
    resumeFile = "/home/nik/uoa/msc-thesis/implementation/checkpoints/test_23.04.17_23:31:25"
    saveFolder = "/home/nik/uoa/msc-thesis/implementation/checkpoints"

    # restore dataset parameters to continue a run
    # this is separate from graph resuming, as the latter requires a session instance
    def resume_metadata(self, dataset):
        if self.do_resume:
            savefile_metapars = self.resumeFile + ".snap"
            print2("Resuming iteration snap from file:")
            print2(savefile_metapars, indent=1)
            try:
                # load saved parameters pickle
                with open(savefile_metapars, 'rb') as f:
                    params = pickle.load(f)
                    dataset.load_metaparams(params)
            except Exception as ex:
                error(ex)

    # restore graph variables to continue a run
    def resume_graph(self, sess):
        if self.do_resume:
            savefile_graph = self.resumeFile + ".graph"
            print2("Resuming run data from file:")
            print2(savefile_graph,indent=1)

            try:
                # load saved graph file
                self.saver.restore(sess, savefile_graph)
            except Exception as ex:
                error(ex)


    # save graph and dataset stuff
    def save(self, sess, dataset,progress):
        try:
            # save the graph
            now = get_datetime_str()
            basename = self.saveFolder + os.path.sep + self.run_id + "_" + progress + "_" + now
            savefile_graph = basename + ".graph"
            savefile_metapars = basename + ".snap"

            print2("Saving graph to [%s]" % savefile_graph, indent=1)
            self.saver.save(sess, savefile_graph)
            # save dataset metaparams
            print2("Saving params to [%s]" % savefile_graph, indent=1)
            params2save = []
            params2save.extend([ dataset.batchIndexTrain, dataset.batchIndexVal])

            with open(savefile_metapars,'wb') as f:
                params = [dataset.batchIndexTrain, dataset.batchIndexVal, dataset.outputFolder]
                pickle.dump(params,f)
        except Exception as ex:
            error(ex)


# dataset class
###############
# Everything concerning the dataset is here
class Dataset:
    useImageMean = False
    num_classes = None
    allImages = []

    # test files
    videoFramePathsFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/test/videoPaths.txt"
    frameClassesFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/test/videoClasses.txt"
    classIndexesNamesFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/test/videoIndexesNames.txt"

    # video -based annotation
    # videoPathsFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/UCF-101_videos_frames16/videoPaths.txt"
    # videoClassesFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/UCF-101_videos_frames16/videoClasses.txt"
    # classIndexesNamesFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/UCF-101_videos_frames16/videoIndexesNames.txt"

    outputFolder = "/home/nik/uoa/msc-thesis/implementation/runData/"
    serializedTrainSetFile = outputFolder + "serializedTrain"
    serializedValSetFile = outputFolder + "serializedVal"
    num_frames_per_video = 16
    imageFormat = "jpg"
    trainValRatio = 2.0 / 3.0

    videoPaths = []
    videoClassNames = []

    meanImage = None
    meanImagePath =  "meanImage_" + str(num_frames_per_video)
    trainSetPath =  "traiset_" + str(num_frames_per_video)
    valSetPath =  "valset_" + str(num_frames_per_video)


    imageShape = (224,224,3)

    # video - based annotation
    # per-class video indexes
    trainPerClass = []
    valPerClass = []
    # total list
    trainSet = []
    trainLabels = []
    valSet = []
    valLabels = []

    phase = None
    # batches
    batchSizeTrain = 3
    batchesTrain = []
    batchIndexTrain = None
    batchConfigFileTrain = "batches_train_sz%d_frv%d" % (batchSizeTrain, num_frames_per_video)
    train_iterator = None

    batchSizeVal = 2
    batchesVal = []
    batchIndexVal = None
    batchConfigFileVal = "batches_val_sz%d_frv%d" % (batchSizeTrain, num_frames_per_video)
    val_iterator = None

    # misc
    verbosity = 1



    # get class name from class index
    def getClassName(self,classIndex):
        return self.videoClassNames[classIndex]

    # read paths to video folders
    def read_video_metadata(self):
        print("Reading video paths from ", self.videoFramePathsFile)
        with open(self.videoFramePathsFile) as f:
            for line in f:
                self.videoPaths.append(line.strip())
        with open(self.classIndexesNamesFile) as f:
            for line in f:
                self.videoClassNames.append(line.strip())
        self.num_classes = len(self.videoClassNames)
    # split to trainval, on a video-based shuffling scheme
    def partition_to_train_val(self):
        if os.path.isfile(self.trainSetPath) \
                and os.path.isfile(self.valSetPath):
            print('Loading training partitioning from file:')
            print2(self.trainSetPath,indent=1)
            with open(self.trainSetPath,'rb') as f:
                self.trainSet, self.trainLabels = pickle.load(f)
            print('Loading validation partitioning from file:')
            print2(self.valSetPath, indent=1)
            with open(self.valSetPath,'rb') as f:
                self.valSet, self.valLabels = pickle.load(f)
            return

        print("Partitioning training/validation sets with a ratio of ",str(self.trainValRatio))
        # read ground truth label per video, make containers:
        #  videos per class histogram
        #
        videoLabels = []
        videoIndexesPerClass = [[] for i in range(self.num_classes)]

        with open(self.frameClassesFile) as f:
            frameidx = 0
            for videoLabel in f:
                label = int(videoLabel)
                videoLabels.append(label)
                videoIndexesPerClass[label].append(frameidx)
                frameidx += 1

        # partition to training & validation
        # for each class
        for cl in range(self.num_classes):
            # shuffle videos and respective labels
            shuffle(videoIndexesPerClass[cl])
            numVideosForClass = len(videoIndexesPerClass[cl])
            numTrain = round(self.trainValRatio * numVideosForClass)

            self.trainPerClass.append(videoIndexesPerClass[cl][:numTrain])
            self.valPerClass.append(videoIndexesPerClass[cl][numTrain:])


        # finally, concat everything,  use temp. containers as we'll shuffle the
        # training data  - no need to shuffle for the validation set though
        totalTrain = []
        totalTrainLabels = []
        for i in range(self.num_classes):
            totalTrain.extend(self.trainPerClass[i])
            totalTrainLabels.extend([i for _ in range(len(self.trainPerClass[i]))])
            self.valSet.extend(self.valPerClass[i])
            self.valLabels.extend([i for _ in range(len(self.valPerClass[i]))])

        # shuffle via the index
        idx = list(range(len(totalTrain)))
        shuffle(idx)
        self.trainSet = [0 for _ in idx] # idx used just for its len
        self.trainLabels = [0 for _ in idx]

        for i in range(len(idx)):
            self.trainSet[i] = totalTrain[idx[i]]
            self.trainLabels[i] = totalTrainLabels[idx[i]]

        # save data
        with open(self.trainSetPath,'wb') as f:
            pickle.dump([self.trainSet, self.trainLabels],f)
        with open(self.valSetPath,'wb') as f:
            pickle.dump([self.valSet, self.valLabels],f)


    # partition to batches
    def calculate_batches(self):
        if not self.batchIndexTrain:
            self.batchIndexTrain = 0
        if not self.batchIndexVal:
            self.batchIndexVal = 0

        # arrange training set to batches
        if os.path.isfile(self.batchConfigFileTrain) \
                and os.path.isfile(self.batchConfigFileVal):
            print('Loading training batches from file:')
            print2(self.batchConfigFileTrain, indent=1)
            with open(self.batchConfigFileTrain, 'rb') as f:
                self.batchesTrain = pickle.load(f)
            print('Loading validation batches from file:')
            print2(self.batchConfigFileVal, indent=1)
            with open(self.batchConfigFileVal, 'rb') as f:
                self.batchesVal = pickle.load(f)
            return


        # training set
        for vididx in range(0, len(self.trainSet), self.batchSizeTrain):
            firstVideoInBatch = vididx
            lastVideoInBatch = min(firstVideoInBatch + self.batchSizeTrain, len(self.trainSet))
            videos = self.trainSet[firstVideoInBatch: lastVideoInBatch]
            lbls = self.trainLabels[firstVideoInBatch: lastVideoInBatch]
            self.batchesTrain.append([videos, lbls])
        print('Calculated ', str(len(self.batchesTrain)), "batches for ", str(len(self.trainSet)), " videos, where ", \
              str(len(self.batchesTrain)), " x ", str(self.batchSizeTrain), " = ",
              str(len(self.batchesTrain) * self.batchSizeTrain), \
              " and #videos = ", str(len(self.trainSet)), ". Last batch has ", str(len(self.batchesTrain[-1][0])))

        # validation set
        for vididx in range(0, len(self.valSet), self.batchSizeVal):
            firstVideoInBatch = vididx
            lastVideoInBatch = min(firstVideoInBatch + self.batchSizeVal, len(self.valSet))
            videos = self.valSet[firstVideoInBatch: lastVideoInBatch]
            lbls = self.valLabels[firstVideoInBatch: lastVideoInBatch]
            self.batchesVal.append([videos, lbls])
        print('Calculated ', str(len(self.batchesVal)), "batches for ", str(len(self.valSet)), " videos, where ", \
              str(len(self.batchesVal)), " x ", str(self.batchSizeVal), " = ",
              str(len(self.batchesVal) * self.batchSizeVal), \
              " and #videos = ", str(len(self.valSet)), ". Last batch has ", str(len(self.batchesVal[-1][0])))

        # save config
        with open(self.batchConfigFileTrain,'wb') as f:
            pickle.dump(self.batchesTrain, f)
        with open(self.batchConfigFileVal,'wb') as f:
            pickle.dump(self.batchesVal, f)

    # write to read images from tensorboard too
    def compute_image_mean(self):
        if os.path.isfile(self.meanImagePath):
            print('Loading mean image from file.')
            with open(self.meanImagePath, 'rb') as f:
                self.meanImage= pickle.load(f)
            return
        meanImage = np.zeros(self.imageShape)
        print("Computing training image mean.")
        for vi in range(len(self.trainSet)):
            print("Video ",str(vi)," / ",str(len(self.trainSet)))
            frames = self.get_video_frames(vi, useMeanCorrection=False)
            for frame in frames:
                meanImage += frame
        self.meanImage = meanImage / (len(self.trainSet) * self.num_frames_per_video)
        with open(self.meanImagePath,'wb') as f:
            pickle.dump(meanImage,f)
        imsave(self.meanImagePath + "." + self.imageFormat, self.meanImage )


    # display image
    def display_image(self,image,label=None):
        print(label)
        plt.title(label)
        plt.imshow(image)
        plt.show()
        # plt.waitforbuttonpress()

    # convert images to  TFRecord
    # ---------------------------
    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def write_images_tfrecord(self, sett, mode="train"):


        if not self.videoPaths:
            error('No video paths stored.')
        # read class per video
        videoClasses = []
        with open(self.frameClassesFile, 'r') as f:
            for line in f:
                line  = line.strip()
                videoClasses.append(int(line))
        if mode == 'train':
            if not os.path.isfile(self.serializedTrainSetFile):
                print('Writing images for mode: [', mode, '] to TFRecord format.')
                self.serialize_to_tfrecord(self.trainSet, self.trainLabels, self.serializedTrainSetFile, "train")
            self.train_iterator = tf.python_io.tf_record_iterator(path=self.serializedTrainSetFile,name = "train_iterator")
        elif mode == 'val':
            if not os.path.isfile(self.serializedValSetFile):
                self.serialize_to_tfrecord(self.valSet, self.valLabels, self.serializedValSetFile, "val")
            self.val_iterator = tf.python_io.tf_record_iterator(path=self.serializedValSetFile,name="val_iterator")
        else:
            error("Use train or val mode for TFRecord serialization. Undefined mode: [%s]" % mode)

    def serialize_to_tfrecord(self, vididxs, labels, outfile, descr=""):

        writer = tf.python_io.TFRecordWriter(outfile)

        count = 0
        for vi in vididxs:
            print(descr,' video ', str(count + 1), " / ", str(len(vididxs)), "  " + self.videoPaths[vi])
            frames = self.get_video_frames(vi,useMeanCorrection=False)
            for frame in frames:

                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': self._int64_feature(self.imageShape[0]),
                    'width': self._int64_feature(self.imageShape[1]),
                    'depth': self._int64_feature(self.imageShape[2]),
                    'label': self._int64_feature(int(labels[count])),
                    'image_raw': self._bytes_feature(frame.tostring())}))
                writer.write(example.SerializeToString())
            count += 1
        writer.close()

        # ---------------------------

    def deserialize_from_tfrecord(self, iterator, numVideos):

        images = []
        labels = []
        for _ in range(numVideos * self.num_frames_per_video):
            try:
                string_record = next(iterator)
                example = tf.train.Example()
                example.ParseFromString(string_record)
                height = int(example.features.feature['height']
                             .int64_list
                             .value[0])
                width = int(example.features.feature['width']
                            .int64_list
                            .value[0])
                img_string = (example.features.feature['image_raw']
                              .bytes_list
                              .value[0])
                depth = (example.features.feature['depth']
                         .int64_list
                         .value[0])
                # label = (example.features.feature['label']
                #          .int64_list
                #          .value[0])
                img_1d = np.fromstring(img_string, dtype=np.uint8)
                # watch it : hardcoding preferd dimensions according to the dataset object.
                # it should be the shape of the stored image instead, for generic use
                image = img_1d.reshape((self.imageShape[0], self.imageShape[1], self.imageShape[2]))

                if self.useImageMean:
                    image = np.ndarray.astype(image, np.float32) - self.meanImage
                images.append(image)
                # labels.append(label)
                # imsave('reconstructedBGR.JPEG', image)
                # image = image[:, :, ::-1] # swap 1st and 3rd dimension
                # imsave('reconstructedBGR__2.JPEG', image)
            except StopIteration:
                break
            except Exception as ex:
                print('Exception at reading image, loading from scratch')
                print(ex)
        # return images, labels
        return images


    # read next batch
    def read_next_batch(self):
        images = []
        if self.phase == "train":
            batches = self.batchesTrain
            batchIndex = self.batchIndexTrain
            iterator = self.train_iterator
            batchSize = self.batchSizeTrain
            self.batchIndexTrain += self.batchSizeTrain
        else:
            batches = self.batchesVal
            batchIndex = self.batchIndexVal
            iterator = self.val_iterator
            batchSize = self.batchSizeVal
            self.batchIndexVal += self.batchSizeVal

        currentBatch = batches[batchIndex]
        if iterator == None:
            # read images from disk
            for vi in currentBatch[0]:
                videoframes = self.get_video_frames(vi,self.useImageMean)
                images.extend(videoframes)
        else:
            # read images from the TFrecord
            images = self.deserialize_from_tfrecord(iterator, batchSize)

        labels = currentBatch[1]
        labels_onehot = labels_to_one_hot(labels, self.num_classes)
        #labels = [ l for l in labels for _ in range(self.num_frames_per_video) ] # duplicate label, er frame

        return images,labels_onehot,labels

    # read all frames for a video
    def get_video_frames(self,videoidx,useMeanCorrection=True):
        print2("Reading frames of video idx %d from disk." % videoidx,req_lvl=1 ,lvl=self.verbosity)
        videoPath = self.videoPaths[videoidx]
        frames = []
        for im in range(self.num_frames_per_video):
            impath = videoPath + os.sep + str(1+im) + "." + self.imageFormat
            frames.append(self.read_image(impath,useMeanCorrection))
        return frames

    # read image from disk
    def read_image(self,imagepath, useMeanCorrection=False):
        image = imread(imagepath)

        print2("Reading image %s" % imagepath, req_lvl=2 ,lvl=self.verbosity)
        # for grayscale images, duplicate
        # intensity to color channels
        if len(image.shape) <= 2:
            image = np.repeat(image[:, :, np.newaxis], 3, 2)
        # drop channels other than RGB
        image = image[:,:,:3]
        #  convert to BGR
        image = image[:, :, ::-1]
        # resize
        image = imresize(image, self.imageShape)

        if useMeanCorrection:
            image = image - self.meanImage
        return image

    # do preparatory work
    def initialize(self, sett):
        self.verbosity = sett.verbosity
        if not sett.do_resume:
            self.outputFolder = self.outputFolder + os.path.sep + sett.run_id + "_" + get_datetime_str()
            print2("Initializing run on folder [%s]" % self.outputFolder)
            # make output dir
            if not os.path.exists(self.outputFolder):
                try:
                    os.makedirs(self.outputFolder)
                except Exception as ex:
                    error(ex)

            self.set_file_paths()

        self.read_video_metadata()
        self.partition_to_train_val()
        self.calculate_batches()
        if self.useImageMean:
            self.compute_image_mean()

        if sett.dataFormat == "TFRecord":
            self.record_iterator = self.write_images_tfrecord(sett, "train")
            self.record_iterator = self.write_images_tfrecord(sett, "val")

        print("Initialized dataset.")
        self.tell()

    # load saved parameters
    def load_metaparams(self, params):
        i = 0
        self.batchIndexTrain = params[i]; i+=1
        self.batchIndexVal = params[i]; i+=1
        self.outputFolder = params[i]; i+=1
        self.set_file_paths()

    # set file paths
    def set_file_paths(self):
        self.batchConfigFileTrain = self.outputFolder + os.path.sep + self.batchConfigFileTrain
        self.batchConfigFileVal = self.outputFolder + os.path.sep + self.batchConfigFileVal

        self.meanImagePath = self.outputFolder + os.path.sep + self.meanImagePath
        self.trainSetPath = self.outputFolder + os.path.sep + self.trainSetPath
        self.valSetPath = self.outputFolder + os.path.sep + self.valSetPath

    # print active settings
    def tell(self):
        print2("Dataset settings:", pr_type="banner-")

        print("TRAIN:")
        print("batch_size: %d\nnum_batches: %d" % (self.batchSizeTrain, len(self.batchesTrain)))

        print("VAL:")
        print("batch_size: %d\nnum_batches: %d" % (self.batchSizeVal, len(self.batchesVal)))
        print();print()

    # get the batch size
    def get_batch_size(self):
        if self.phase == "train":
            return self.batchSizeTrain
        else:
            return self.batchSizeVal
# LRCN class
############
# Despite its name, this class holds all instances of networks for each run type. splitting it is a far away TODO
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
    # let there be network
    def create(self, settings, dataset, run_mode):

        batchLabelsShape = [None, dataset.num_classes]
        self.inputLabels = tf.placeholder(tf.int32, batchLabelsShape, name="input_labels")
        weightsFile = "/home/nik/Software/tensorflow-tutorials/alexnet/models/alexnet_converted/bvlc_alexnet.npy"

        framesLogits = None

        if run_mode == settings.BASELINE:
            # single DCNN, classifying individual frames
            self.inputData, framesLogits = alexnet.define(dataset.imageShape, weightsFile, dataset.num_classes)
        elif run_mode == settings.LSTM:
            #  DCNN for frame encoding
            self.inputData, self.outputTensor = alexnet.define(dataset.imageShape, weightsFile, dataset.num_classes,settings.lstm_input_layer)
            # LSTM for frame sequence classification for frame encoding
            framesLogits = lstm.define(self.outputTensor, dataset.num_classes)
            print2("Outputs shape:" + str(self.outputTensor.shape), req_lvl=1 ,lvl=dataset.verbosity)

        else:
            error("Unknown run mode [%s]" % run_mode)

        # print2("Inputs shape:" + str(self.inputData.shape), req_lvl=1 ,lvl=settings.verbosity)
        # print2("Input labels shape:" + str(self.inputLabels.shape), req_lvl=1 ,lvl=settings.verbosity)
        # print2("Frame-Logits shape:" + str(framesLogits.shape), req_lvl=1 ,lvl=settings.verbosity)


        # average the logits on the frames dimension
        with tf.name_scope("video_level_pooling_train"):
            # -1 on the number of videos (batchsize) to deal with varying values for test and train
            frameLogits = tf.reshape(framesLogits,(-1, dataset.num_frames_per_video,dataset.num_classes),name = "reshape_framelogits_pervideo")
            self.logits = tf.scalar_mul(1/dataset.num_frames_per_video, tf.reduce_sum(frameLogits, axis=1))

        # loss
        with tf.name_scope("cross_entropy_loss"):
            loss_per_vid = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputLabels, name="loss")
            with tf.name_scope('total'):
                self.loss = tf.reduce_mean(loss_per_vid)

            train_summaries_list.append(add_descriptive_summary(self.loss))

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

        train_summaries_list.append(tf.summary.scalar('accuracyTrain', self.accuracyTrain))
        val_summaries_list.append(tf.summary.scalar('accuracyVal', self.accuracyVal))


# the main function
##################
def main():
    # argparse is quite overkill, but future proof
    parser = argparse.ArgumentParser(description="Run the activity recognition task.")
    parser.add_argument("run_mode", metavar='mode', type=str,
                    help='an integer for the accumulator')
    args = parser.parse_args()

    print2('Running the activity recognition task in mode: [%s]' % args.run_mode, pr_type="banner")
    print() # newline from Lidl

    # create and initialize settings and dataset objects
    settings = Settings()
    dataset = Dataset()
    # resume first
    settings.resume_metadata(dataset)
    # initialize & pre-process dataset; checks and respects resumed vars
    dataset.initialize(settings)
    dataset.phase = "train"
    # create and configure the nets : CNN and / or lstm
    lrcn = LRCN()
    lrcn.create(settings, dataset, args.run_mode)

    # view_print_tensors(lrcn,dataset,settings,lrcn.print_tensors)

    # create and init. session and visualization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # restore graph variables,
    settings.resume_graph(sess)

    # mop up all summaries. Unless you separate val and train, the training subgraph
    # will be executed when calling the summaries op. Which is bad.
    train_summaries_merged = tf.summary.merge(train_summaries_list)
    val_summaries_merged = tf.summary.merge(val_summaries_list)

    # create the writer for visualizashuns
    tboard_writer = tf.summary.FileWriter(settings.tensorboard_folder, sess.graph)

    # train. Specifying validation and training as separate functions, classes or sth is an essential TODO



    trainBatchCount = 0
    for epochIdx in range(settings.epochs):
        # iterate over the dataset. batchsize refers tp the number of videos
        # frames in batch is batchsize x numFramesperVideo

        print2('Epoch %d / %d ' % (1 + epochIdx, settings.epochs), pr_type="banner")

        # loop for all batches in the epoch. Noobish, rework needed in line of the above TODO
        while dataset.batchIndexTrain <= len(dataset.batchesTrain):
            print2("Batch %d / %d " % (1+dataset.batchIndexTrain , len(dataset.batchesTrain)))
            # read  batch
            images, labels_onehot, labels = dataset.read_next_batch()

            # feed the batch through the network to get activations per frame and run optimizer
            print("Running %s batch of %d images, %d labels: %s , %s." % (dataset.phase, len(images) ,len(labels_onehot), str(labels[0]),str(labels_onehot[0])))

            logits = sess.run(lrcn.logits,feed_dict={lrcn.inputData:images, lrcn.inputLabels:labels_onehot})
            print(logits)
            summaries_train, batch_loss, _, accuracy = sess.run([train_summaries_merged, lrcn.loss , lrcn.optimizer, lrcn.accuracyTrain], feed_dict={lrcn.inputData:images, lrcn.inputLabels:labels_onehot})
            print("Train accuracy: ", accuracy)
            print ("Epoch %2d, batch %2d / %2d, loss: %4.3f" % ( epochIdx, dataset.batchIndexTrain, len(dataset.batchesTrain), batch_loss ))


            # test every half epoch ?

            if trainBatchCount == math.floor(len(dataset.batchesTrain)/2) or True:
                valBatchCount = 0
                dataset.phase = "val"
                # validation
                while dataset.batchIndexVal <= len(dataset.batchesVal) or True:


                    images, labels_onehot, labels = dataset.read_next_batch()

                    print("Running %s batch of %d images, %d labels: %s , %s." % (
                    dataset.phase, len(images), len(labels_onehot), str(labels[0]), str(labels_onehot[0])))

                    summaries_val, accuracy = sess.run([val_summaries_merged, lrcn.accuracyVal], feed_dict={lrcn.inputData: images, lrcn.inputLabels : labels_onehot})
                    print("Validation accuracy: ", accuracy)
                    tboard_writer.add_summary(summaries_val, global_step=epochIdx * len(dataset.batchesVal )  + trainBatchCount)


            # pass stuff to tensorboard and write
            tboard_writer.add_summary(summaries_train,global_step=trainBatchCount)

            tboard_writer.flush()
            trainBatchCount += 1
        # save a checkpoint every epoch
        settings.save(sess,dataset,progress = "ep_%d_btch_%d" % (epochIdx, trainBatchCount))

    # mop up
    tboard_writer.close()
    sess.close()

if __name__ == "__main__":
    main()
