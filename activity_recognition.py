import tensorflow as tf
from random import shuffle

import os
from scipy.misc import imread, imresize, imsave # TODO remove imsage
import numpy as np
import matplotlib.pyplot as plt
import pickle

from models.alexnet import alexnet
from models.lstm import lstm
# settings class
class Settings:
    networkName = "alexnet"
    epochs = 10
    batchSize = 128
    tensorboard_folder = "/home/nik/uoa/msc-thesis/implementation/tensorboard_graphs"

    dataFormat = "TFRecord" # TFRecord or raw


    #dataFormat = "raw"  # TFRecord or raw


# dataset class
class Dataset:
    useImageMean = False
    numClasses = 101
    allImages = []
    # video -based annotation
    videoPathsFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/UCF-101_videos_frames16/videoPaths.txt"
    videoClassesFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/UCF-101_videos_frames16/videoClasses.txt"
    classIndexesNamesFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/UCF-101_videos_frames16/videoIndexesNames.txt"

    outputFolder = "/home/nik/uoa/msc-thesis/datasets/UCF101/UCF-101_videos_frames16/"
    serializedTrainSetFile = outputFolder + "serializedTrain"
    serializedValSetFile = outputFolder + "serializedVal"
    numFramesPerVideo = 16
    imageFormat = "jpg"
    trainValRatio = 2.0 / 3.0

    videoPaths = []
    videoClassNames = []

    meanImage = None
    meanImagePath = outputFolder + "/meanImage_" + str(numFramesPerVideo)
    trainSetPath = outputFolder + "/traiset_" + str(numFramesPerVideo)
    valSetPath = outputFolder + "/valset_" + str(numFramesPerVideo)
    batchConfigPathTrain = outputFolder + "/batchPlanTrain_" + str(numFramesPerVideo)
    batchConfigPathVal = outputFolder + "/batchPlanVal_" + str(numFramesPerVideo)

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

    # batches
    batchSizeTrain = 1
    batchesTrain = []
    batchIndexTrain = None
    batchSizeVal = 1
    batchesVal = []
    batchIndexVal = None
    train_iterator = None  # to read serialized image data
    val_iterator = None  # to read serialized image data

    # get class name from class index
    def getClassName(self,classIndex):
        return self.videoClassNames[classIndex]

    # read paths to video folders
    def read_video_metadata(self):
        print("Reading video paths from ",self.videoPathsFile)
        with open(self.videoPathsFile) as f:
            for line in f:
                self.videoPaths.append(line.strip())
        with open(self.classIndexesNamesFile) as f:
            for line in f:
                self.videoClassNames.append(line.strip())

    # split to trainval, on a video-based shuffling scheme
    def partition_to_train_val(self):
        if os.path.isfile(self.trainSetPath) \
                and os.path.isfile(self.valSetPath) \
                and os.path.isfile(self.batchConfigPathTrain) \
                and os.path.isfile(self.batchConfigPathVal):
            print('Loading training/validation partition from file.')
            with open(self.trainSetPath,'rb') as f:
                self.trainSet, self.trainLabels = pickle.load(f)
            with open(self.valSetPath,'rb') as f:
                self.valSet, self.valLabels = pickle.load(f)
            with open(self.batchConfigPathTrain,'rb') as f:
                self.batchesTrain = pickle.load(f)
            with open(self.batchConfigPathVal,'rb') as f:
                self.batchesVal = pickle.load(f)
            return

        print("Partitioning training/validation sets with a ratio of ",str(self.trainValRatio))
        # read ground truth label per video, make containers:
        #  videos per class histogram
        #
        videoLabels = []
        videoIndexesPerClass = [[] for i in range(self.numClasses)]

        with open(self.videoClassesFile) as f:
            videoidx = 0
            for videoLabel in f:
                label = int(videoLabel)
                videoLabels.append(label)
                videoIndexesPerClass[label].append(videoidx)
                videoidx += 1

        # partition to training & validation
        # for each class
        for cl in range(self.numClasses):
            # shuffle videos and respective labels
            shuffle(videoIndexesPerClass[cl])
            numVideosForClass = len(videoIndexesPerClass[cl])
            numTrain = round(self.trainValRatio * numVideosForClass)
            numVal = numVideosForClass - numTrain

            self.trainPerClass.append(videoIndexesPerClass[cl][:numTrain])
            self.valPerClass.append(videoIndexesPerClass[cl][numTrain:])


        # finally, concat everything,  use temp. containers as we'll shuffle the
        # training data  - no need to shuffle for the validation set though
        totalTrain = []
        totalTrainLabels = []
        for i in range(self.numClasses):
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


        # arrange training set to batches
        # training set
        for vididx in range(0, len(self.trainSet), self.batchSizeTrain):
            firstVideoInBatch = vididx
            lastVideoInBatch = min(firstVideoInBatch + self.batchSizeTrain, len(self.trainSet))
            videos = self.trainSet[firstVideoInBatch : lastVideoInBatch]
            lbls = self.trainLabels[firstVideoInBatch : lastVideoInBatch]
            self.batchesTrain.append([videos, lbls])
        print('Calculated ', str(len(self.batchesTrain)), "batches for ", str(len(self.trainSet)), " videos, where ", \
              str(len(self.batchesTrain)), " x ", str(self.batchSizeTrain), " = ", str(len(self.batchesTrain) * self.batchSizeTrain), \
              " and #videos = ", str(len(self.trainSet)), ". Last batch has ", str(len(self.batchesTrain[-1][0])))

        # validation set
        for vididx in range(0, len(self.valSet), self.batchSizeVal):
            firstVideoInBatch = vididx
            lastVideoInBatch = min(firstVideoInBatch + self.batchSizeVal, len(self.valSet))
            videos = self.valSet[firstVideoInBatch : lastVideoInBatch]
            lbls = self.valLabels[firstVideoInBatch : lastVideoInBatch]
            self.batchesVal.append([videos, lbls])
        print('Calculated ', str(len(self.batchesVal)), "batches for ", str(len(self.valSet)), " videos, where ", \
              str(len(self.batchesVal)), " x ", str(self.batchSizeVal), " = ", str(len(self.batchesVal) * self.batchSizeVal), \
              " and #videos = ", str(len(self.valSet)), ". Last batch has ", str(len(self.batchesVal[-1][0])))


        # save data
        with open(self.trainSetPath,'wb') as f:
            pickle.dump([self.trainSet, self.trainLabels],f)
        with open(self.valSetPath,'wb') as f:
            pickle.dump([self.valSet, self.valLabels],f)
        with open(self.batchConfigPathTrain,'wb') as f:
            pickle.dump(self.batchesTrain, f)
        with open(self.batchConfigPathVal,'wb') as f:
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
        self.meanImage = meanImage / ( len(self.trainSet) * self.numFramesPerVideo)
        with open(self.meanImagePath,'wb') as f:
            pickle.dump(meanImage,f)
        imsave(self.meanImagePath + "." + self.imageFormat, self.meanImage )


    # display image
    def display_image(self,image,label=''):
        print(label, " ", self.videoClassNames[int(label)])
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
        print('Writing images for mode: [',mode,'] to TFRecord format.')

        if not self.videoPaths:
            print('No video paths stored.')
            exit (1)
        # read class per video
        videoClasses = []
        with open(self.videoClassesFile,'r') as f:
            for line in f:
                line  = line.strip()
                videoClasses.append(int(line))
        if mode == 'train':
            if not os.path.isfile(self.serializedTrainSetFile):
                self.serialize_to_tfrecord(self.trainSet, self.trainLabels, self.serializedTrainSetFile, "train")
            self.train_iterator = tf.python_io.tf_record_iterator(path=self.serializedTrainSetFile)
        elif mode == 'val':
            if not os.path.isfile(self.serializedValSetFile):
                self.serialize_to_tfrecord(self.valSet, self.valLabels, self.serializedValSetFile, "val")
            self.val_iterator = tf.python_io.tf_record_iterator(path=self.serializedValSetFile)
        else:
            print("Use train or val mode for TFRecord serialization. Undefined mode :",mode)
            exit(1)

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
        for _ in range(numVideos * self.numFramesPerVideo):
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
    def readNextBatch(self,trainMode=True):
        if not self.batchIndexTrain:
            self.batchIndexTrain = 0
        images = []
        if trainMode:
            batches = self.batchesTrain
            batchIndex = self.batchIndexTrain
            iterator = self.train_iterator
            batchSize = self.batchSizeTrain

        else:
            batches = self.batchesVal
            batchIndex = self.batchIndexVal
            iterator = self.val_iterator
            batchSize = self.batchSizeVal
        currentBatch = batches[batchIndex]
        if iterator == None:
            # read images from disk
            for vi in currentBatch[0]:
                videoframes = self.get_video_frames(vi,self.useImageMean)
                images.extend(videoframes)
        else:
            # read images from the TFrecord
            images = self.deserialize_from_tfrecord(iterator, batchSize)

        labels = [ l for l in currentBatch[1] for _ in range(self.numFramesPerVideo) ] # duplicate label, er frame
        return images,labels

    # read all frames for a video
    def get_video_frames(self,videoidx,useMeanCorrection=True):
        videoPath = self.videoPaths[videoidx]
        frames = []
        for im in range(self.numFramesPerVideo):
            impath = videoPath + os.sep + str(1+im) + "." + self.imageFormat
            frames.append(self.read_image(impath,useMeanCorrection))
        return frames

    # read image from disk
    def read_image(self,imagepath, useMeanCorrection=False):
        image = imread(imagepath)
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
        self.read_video_metadata()
        self.partition_to_train_val()
        if self.useImageMean:
            self.compute_image_mean()

        if sett.dataFormat == "TFRecord":
            self.record_iterator = self.write_images_tfrecord(sett, "train")
            self.record_iterator = self.write_images_tfrecord(sett, "val")

# class Network:
#     inputTensor = None
#     outputTensor = None
#     prob = None
#
#     def define(self,dataset,settings,networkName):
#
#
#         print("Loading network: [",networkName, "]")
#
#         if networkName == "alexnet":
#             # specify dimensions
#             imageW = dataset.imageShape[1]
#             imageH = dataset.imageShape[0]
#             imageD = dataset.imageShape[2]
#             train_x = np.zeros((1, imageW, imageH, imageD)).astype(np.float32)
#             xdim = train_x.shape[1:]
#             inputTensor = tf.placeholder(tf.float32, (None,) + xdim)
#             weightsFile = "/home/nik/Software/tensorflow-tutorials/alexnet/models/alexnet_converted/bvlc_alexnet.npy"
#             self.inputTensor, self.outputTensor = alexnet.define(xdim, weightsFile)
#         elif networkName == "lstm":
#             if self.inputTensor == None:
#                 print("Input tensor not set.")
#                 exit(1)
#             print(self.inputTensor.shape)
#             self.prob = lstm.define_lstm(self.outputTensor)
#         else:
#
#            print("Undefined network ", networkName)
#            return

class LRCN:
    def create(self):
        # specify dimensions
        imageW = dataset.imageShape[1]
        imageH = dataset.imageShape[0]
        imageD = dataset.imageShape[2]
        train_x = np.zeros((1, imageW, imageH, imageD)).astype(np.float32)
        xdim = train_x.shape[1:]
        inputTensor = tf.placeholder(tf.float32, (None,) + xdim)
        weightsFile = "/home/nik/Software/tensorflow-tutorials/alexnet/models/alexnet_converted/bvlc_alexnet.npy"
        self.inputTensor, self.outputTensor = alexnet.define(xdim, weightsFile)
        self.prob = lstm.define_lstm(self.outputTensor)

# run the process
##################

# create and initialize settings and dataset objects
settings = Settings()
dataset = Dataset()
# initialize & pre-process dataset
dataset.initialize(settings)

# create and configure the CNN and the lstm
lrcn = LRCN()
lrcn.create()
# dcnn = Network()
# dcnn.define(dataset,settings,"alexnet")
# group up activations into list of tensors

# create and configure the lstm
# lstm = Network()
# lstm.inputTensor = dcnn.outputTensor
#lstm.define(dataset,settings,"lstm")

# specify loss and optimization
# how is the loss specified for the average pooling case??
#loss =
# create and init. session and visualization
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tboard_writer = tf.summary.FileWriter(settings.tensorboard_folder, sess.graph)


# train
for epochIdx in range(settings.epochs):
    # iterate over the dataset. batchsize refers tp the number of videos
    # frames in batch is batchsize x numFramesperVideo
    # eval all, average pool after eveal
    # OR add it at the network ?
    # read  batch
    images, labels = dataset.readNextBatch(settings)

    # feed the batch through the network to get activations per frame
    inputTensor = network.inputTensor
    sess.run(network.prob, feed_dict={inputTensor:images})
    c=0
    for im in images:
        dataset.display_image(im,label=str(labels[c]))
        c+=1


