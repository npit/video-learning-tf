import tensorflow as tf
from random import shuffle

import os
from scipy.misc import imread, imresize, imsave # TODO remove imsage
import numpy as np
import matplotlib.pyplot as plt
import pickle

from models.alexnet import alexnet

# settings class
class Settings:
    networkName = "alexnet"
    epochs = 10
    batchSize = 128
    tensorboard_folder = "/home/nik/uoa/msc-thesis/implementation/tensorboard_graphs"

    dataFormat = "TFRecord" # TFRecord or raw

# dataset class
class Dataset:
    useImageMean = True
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

    meanImage = None
    meanImagePath = outputFolder + "/meanImage_" + str(numFramesPerVideo)
    trainSetPath = outputFolder + "/traiset_" + str(numFramesPerVideo)
    valSetPath = outputFolder + "/valset_" + str(numFramesPerVideo)
    batchConfigPath = outputFolder + "/batchPlan_" + str(numFramesPerVideo)

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

    # batch
    batchSize = 32
    batches = []
    batchIndex = None
    train_iterator = None  # to read serialized image data
    val_iterator = None  # to read serialized image data

    # read next batch
    def readNextBatch(self,trainMode=True):
        if not self.batchIndex:
            self.batchIndex = 0
            self.imageIndex = 0
        images = []
        labels = []
        if trainMode:
            if self.train_iterator == None:
                # read images from disk
                for vi in self.batches[self.batchIndex]:
                    videoframes = self.get_video_frames(vi)
                    images.append(videoframes)

    # read paths to video folders
    def read_video_paths(self):
        print("Reading video paths from ",self.videoPathsFile)
        with open(self.videoPathsFile) as f:
            for line in f:
                self.videoPaths.append(line.strip())

    # split to trainval, on a video-based shuffling scheme
    def partition_to_train_val(self):
        if os.path.isfile(self.trainSetPath) \
                and os.path.isfile(self.valSetPath) \
                and os.path.isfile(self.batchConfigPath):
            print('Loading training/validation partition from file.')
            with open(self.trainSetPath,'rb') as f:
                self.trainSet, self.trainLabels = pickle.load(f)
            with open(self.valSetPath,'rb') as f:
                self.valSet, self.valLabels = pickle.load(f)
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
        for vididx in range(0,len(self.trainSet), self.batchSize):
            firstVideoInBatch = vididx
            lastVideoInBatch = min(firstVideoInBatch + self.batchSize, len(self.trainSet))
            videos = self.trainSet[firstVideoInBatch : lastVideoInBatch]
            lbls = self.trainLabels[firstVideoInBatch : lastVideoInBatch]
            self.batches.append([videos,lbls])
        print('Calculated ',str(len(self.batches)), "batches for ",str(len(self.trainSet))," videos, where ", \
              str(len(self.batches))," x ",str(self.batchSize)," = ",str(len(self.batches) * self.batchSize), \
              " and #videos = ",str(len(self.trainSet)), ". Last batch has ",str(len(self.batches[-1])))


        # save data
        with open(self.trainSetPath,'wb') as f:
            pickle.dump([self.trainSet, self.trainLabels],f)
        with open(self.valSetPath,'wb') as f:
            pickle.dump([self.valSet, self.valLabels],f)
        with open(self.batchConfigPath,'wb') as f:
            pickle.dump(self.batches,f)


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
            frames = self.get_video_frames(vi)
            for frame in frames:
                meanImage += frame
        self.meanImage = meanImage / ( len(self.trainSet) * self.numFramesPerVideo)
        with open(self.meanImagePath,'wb') as f:
            pickle.dump(meanImage,f)
        #self.display_image(meanImage)

    # read image from disk
    def read_image(self,imagepath, computingImageMean=False):
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

        if not computingImageMean and self.useImageMean:
            image = image - self.meanImage

        return image

    # display image
    def display_image(self,image):
        plt.imshow(image)
        plt.show()
        plt.waitforbuttonpress()

    # convert images to  TFRecord
    # ---------------------------
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def write_images_TFRecord(self, sett,mode="train"):
        print('Writing images in TFRecord format.')
        if os.path.isfile(self.serializedTrainSetFile) and os.path.isfile(self.serializedValSetFile):
            return
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
            self.train_iterator = self.serialize_to_tfrecord(self.trainSet, self.trainLabels, self.serializedTrainSetFile, "train")
        elif mode == 'val':
            self.val_iterator = self.serialize_to_tfrecord(self.valSet, self.valLabels, self.serializedValSetFile, "val")
        else:
            print("Use train or val mode for TFRecord serialization. Undefined mode :",mode)
            exit(1)


    def serialize_to_tfrecord(self, imgidxs, labels, outfile, descr=""):

        writer = tf.python_io.TFRecordWriter(outfile)

        count = 0
        for vi in imgidxs:
            print(descr,' ', str(count + 1), " / ", str(len(imgidxs)), "  " + self.videoPaths[vi])
            frames = self.get_video_frames(vi)
            for frame in frames:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': self._int64_feature(self.imageShape[0]),
                    'width': self._int64_feature(self.imageShape[0]),
                    'depth': self._int64_feature(self.imageShape[0]),
                    'label': self._int64_feature(int(labels[count])),
                    'image_raw': self.bytes_feature(frame.tostring())}))
                writer.write(example.SerializeToString())
                count += 1
        writer.close()
        return tf.python_io.tf_record_iterator(path=self.outfile)
        # ---------------------------

    def get_video_frames(self,videoidx):
        videoPath = self.videoPaths[videoidx]
        frames = []
        for im in range(self.numFramesPerVideo):
            impath = videoPath + os.sep + str(1+im) + "." + self.imageFormat
            frames.append(self.read_image(impath,computingImageMean=True))
        return frames

    def initialize(self, sett):
        self.read_video_paths()
        self.partition_to_train_val()
        if self.useImageMean:
            self.compute_image_mean()

        if sett.dataFormat == "TFRecord":
            self.record_iterator = self.write_images_TFRecord(sett,"train")
            self.record_iterator = self.write_images_TFRecord(sett,"val")

class Network:
    inputTensor = None
    prob = None
    weightsFile = "/home/nik/Software/tensorflow-tutorials/alexnet/models/alexnet_converted/bvlc_alexnet.npy"

    def define(self,dataset,settings):
        # specify dimensions
        imageW = dataset.imageShape[1]
        imageH = dataset.imageShape[0]
        imageD = dataset.imageShape[2]
        train_x = np.zeros((1, imageW, imageH, imageD)).astype(np.float32)
        xdim = train_x.shape[1:]
        inputTensor = tf.placeholder(tf.float32, (None,) + xdim)

        print("Loading network: [",settings.networkName, "]")

        if settings.networkName == "alexnet":
           inputTensor, prob = alexnet.define(xdim, self.weightsFile)
        else:

           print("Undefined network ", settings.networkName)
           return




# run the process
##################

# define and partition the training data
settings = Settings()
dataset = Dataset()

dataset.initialize(settings)

network = Network()

network.define(dataset,settings)

sess = tf.Session()
tboard_writer = tf.summary.FileWriter(settings.tensorboard_folder, sess.graph)

# train
for epochIdx in range(settings.epochs):
    # iterate over the dataset. batchsize refers tp the number of videos
    # frames in batch is batchsize x numFramesperVideo
    # eval all, average pool after eveal
    # OR add it at the network ?
    # read  batch
     images, labels = dataset.readBatch()


