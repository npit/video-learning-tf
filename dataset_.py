
# files and io
import os
import pickle
# utils
from utils_ import *
from random import shuffle
# image IO
from scipy.misc import imread, imresize, imsave
# displaying
import matplotlib.pyplot as plt


class Dataset:
    useImageMean = False
    num_classes = None


    # dataset discovery parameters
    # consequtive video frames mode
    ##########################
    # test files, much smaller than orig

    videoFramePathsFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/test/videoPaths.txt"
    frameClassesFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/test/videoClasses.txt"

    # video -based annotation
    # videoPathsFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/UCF-101_videos_frames16/videoPaths.txt"
    # videoClassesFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/UCF-101_videos_frames16/videoClasses.txt"
    # classIndexesNamesFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/UCF-101_videos_frames16/videoIndexesNames.txt"

    #############################

    # non  consequtive video frames mode: pretty much image reconition
    # caffe-like input file

    frames_labels_train = "/home/nik/uoa/msc-thesis/implementation/frames.train"
    frames_labels_test  = "/home/nik/uoa/msc-thesis/implementation/frames.test"

    frame_paths = []
    frame_classes = []
    # file to map class indexes to their names
    classIndexesNamesFile = "/home/nik/uoa/msc-thesis/datasets/UCF101/test/videoIndexesNames.txt"



    #
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

    input_mode = None
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
    epochIndex = None
    epochs = None
    # batches
    batchCount = 0

    batchSize = None
    batchIndex = 0
    batches = None
    iterator = None

    batchSizeTrain = 3
    batchesTrain = []
    batchIndexTrain = None
    batchConfigFileTrain = "batches_train_sz%d_frv%d" % (batchSizeTrain, num_frames_per_video)
    train_iterator = None

    do_validation = False
    validation_interval = 1
    batchSizeVal = 10
    batchesVal = []
    batchIndexVal = None
    batchConfigFileVal = "batches_val_sz%d_frv%d" % (batchSizeTrain, num_frames_per_video)
    val_iterator = None

    # misc
    verbosity = 1
    logger = None

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

    # read paths to images
    def read_frames_metadata(self):
        # store train and test in same variable
        self.frame_paths.extend([[],[]])
        self.frame_classes.extend([[], []])
        with open(self.frames_labels_train,'r') as f:
            for im in f:
                impath,imlabel = im.split()
                self.frame_paths[defs.phase.train].append(impath)
                self.frame_classes[defs.phase.train].append(imlabel)

            with open(self.frames_labels_test, 'r') as f:
                for im in f:
                    impath, imlabel = im.split()
                    self.frame_paths[defs.phase.val].append(impath)
                    self.frame_classes[defs.phase.val].append(imlabel)

        # this assumes that instances for all classes are provided - crashy down the road but nice error check
        self.num_classes = len(set(self.frame_classes[defs.phase.train]))

    # split to trainval, on a video-based shuffling scheme
    def partition_to_train_val_videowise(self):
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
    def calculate_batches_videowise(self):
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
    def compute_image_mean_videowise(self):
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

    # helper tfrecord function
    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # helper tfrecord function
    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # convert images to  TFRecord
    def write_images_tfrecord_videowise(self, sett, mode="train"):


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

    def serialize_to_tfrecord_videowise(self, vididxs, labels, outfile, descr=""):

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

    def deserialize_from_tfrecord(self, iterator, images_per_iteration):
        # images_per_iteration :
        images = []
        labels = []
        for _ in range(images_per_iteration * self.num_frames_per_video):
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
    def read_next_batch_videowise(self):
        images = []
        currentBatch = self.batches[self.batchIndex]
        if self.iterator == None:
            # read images from disk
            for vi in currentBatch[0]:
                videoframes = self.get_video_frames(vi, self.useImageMean)
                images.extend(videoframes)
        else:
            # read images from the TFrecord
            images = self.deserialize_from_tfrecord(self.iterator, self.batchSize)

        labels = currentBatch[1]
        labels_onehot = labels_to_one_hot(labels, self.num_classes)
        # labels = [ l for l in labels for _ in range(self.num_frames_per_video) ] # duplicate label, er frame

        return images, labels_onehot#, labels

    def read_next_batch(self):
        if self.input_mode == "videowise":
            images, labels =  self.read_next_batch_videowise()
        else:
            images, labels = self.read_next_batch_framewise()

        self.batchIndex = self.batchIndex + 1
        return images, labels

    def set_phase(self, phase):
        # if the network is switching phases, store the index
        # to continue later
        if self.phase is not None:
            print2("Suspending phase [%s], on current batch index [%d]" % (defs.phase.str(self.phase),
                                                                           self.batchIndex), req_lvl=1,
                   lvl=self.verbosity)
            if self.phase == defs.phase.train:
                self.batchIndexTrain = self.batchIndex
            else:
                self.batchIndexVal = self.batchIndex

        print2("Setting phase [%s], on batch index [%d]" % (defs.phase.str(phase),
                                                                        self.batchIndexVal), req_lvl=1, lvl=self.verbosity)
        # update phase variables
        self.phase = phase
        if self.phase == defs.phase.train:
            self.batches = self.batchesTrain
            self.batchIndex = self.batchIndexTrain
            self.iterator = self.train_iterator
            self.batchSize = self.batchSizeTrain
        else:
            self.batches = self.batchesVal
            self.batchIndex = self.batchIndexVal
            self.iterator = self.val_iterator
            self.batchSize = self.batchSizeVal

    def read_next_batch_framewise(self):
        images = []
        currentBatch = self.batches[self.batchIndex]
        if self.iterator == None:
            # read images from disk
            for impath in currentBatch[0]:
                frame = self.read_image(impath,self.useImageMean)
                images.append(frame)
        else:
            # read images from the TFrecord
            images = self.deserialize_from_tfrecord(self.iterator, 1)

        labels = currentBatch[1]
        labels_onehot = labels_to_one_hot(labels, self.num_classes)
        # labels = [ l for l in labels for _ in range(self.num_frames_per_video) ] # duplicate label, er frame

        return images, labels_onehot#, labels


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
        self.epochs = sett.epochs
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
            self.epochIndex = 0
            self.batchIndexTrain = 0
            self.batchIndexVal = 0

        self.input_mode = sett.input_mode
        if self.input_mode ==  defs.input_mode.video:
            self.initialize_videowise(sett)
        elif self.input_mode ==  defs.input_mode.image:
            self.initialize_framewise(sett)
        else:
            error("Undefined frame format : %s " % sett.frame_format)
        self.logger = sett.logger
        print("Initialized dataset.")
        self.tell()

    # video-wise initialization
    def initialize_videowise(self, sett):
        # here frames are fetched video-wise, i.e. all video frames will be in order for a given video
        self.read_video_metadata()
        self.partition_to_train_val_videowise()
        self.calculate_batches_videowise()
        if self.useImageMean:
            self.compute_image_mean_videowise()

        if sett.dataFormat == "TFRecord":
            self.record_iterator = self.write_images_tfrecord_videowise(sett, "train")
            self.record_iterator = self.write_images_tfrecord_videowise(sett, "val")

    # framewise initialization
    def initialize_framewise(self,sett):
        self.read_frames_metadata()
        self.batchesTrain = []
        self.batchesVal = []

        # train
        imgs = sublist(self.frame_paths[defs.phase.train], self.batchSizeTrain)
        lbls = sublist(self.frame_classes[defs.phase.train], self.batchSizeTrain)
        for l in range(len(lbls)):
            self.batchesTrain.append([
                imgs[l],
                list(map(int, lbls[l]))
                     ])
        # val
        imgs = sublist(self.frame_paths[defs.phase.val], self.batchSizeVal)
        lbls = sublist(self.frame_classes[defs.phase.val], self.batchSizeVal)
        for l in range(len(lbls)):
            self.batchesVal.append([
                imgs[l],
                list(map(int, lbls[l]))
                     ])

        if sett.frame_format == "TFRecord":
            error("TODO tfrecord framewise")
        pass

    # load saved parameters
    def load_metaparams(self, params):
        i = 0
        self.batchIndexTrain = params[i]; i+=1
        self.batchIndexVal = params[i]; i+=1
        self.outputFolder = params[i]; i+=1
        self.epochIndex = params[i]; i+=1
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
        print2("Dataset batch information per run mode:",pr_type="banner-")
        print("%-8s %-6s %-6s %-6s" % ("Mode","size","num","index"))
        print("%-8s %-6d %-6d %-6d" %
              (defs.phase.str(defs.phase.train), self.batchSizeTrain, len(self.batchesTrain), self.batchIndexTrain))
        print("%-8s %-6d %-6d %-6d" %
              (defs.phase.str(defs.phase.val), self.batchSizeVal, len(self.batchesVal), self.batchIndexVal))
        print();print()

    # get the batch size
    def get_batch_size(self):
        if self.phase == "train":
            return self.batchSizeTrain
        else:
            return self.batchSizeVal

    # get the global step
    def get_global_step(self):
        if self.phase == 'train':
            return self.epochIndex * self.batchSizeTrain + self.batchIndexTrain
        else:
            return self.epochIndex * self.batchSizeVal + self.batchIndexVal

    # print iteration information
    def print_iter_info(self, num_images, num_labels):
        print2("%s Batch %4d / %4d : %3d images, %3d labels" % (
        defs.phase.str(self.phase), self.batchIndex, len(self.batches), num_images, num_labels))

    # specify valid loop iteration
    def loop(self):
        return self.batchIndex < len(self.batches)

    def should_test_now(self):
        return self.do_validation and (self.batchCount // self.validation_interval == 0 )