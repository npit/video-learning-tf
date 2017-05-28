
# files and io

import os, pickle

# utils
from utils_ import *
from random import choice, randrange
# image IO
from scipy.misc import imread, imresize, imsave
# displaying
import matplotlib.pyplot as plt

# Dataset class
class Dataset:

    num_classes = None

    # input file names
    input_source_files = [[], []]  # train,test

    # video frame paths and classes
    video_shots = []
    video_classes = []

    # single frame paths and classes, for raw input mode
    frame_paths = []
    frame_classes = []

    # file to map class indexes to their names
    class_names = "/home/nik/uoa/msc-thesis/datasets/UCF101/test/videoIndexesNames.txt"

    #
    run_folder = "/home/nik/uoa/msc-thesis/implementation/runData/"
    serializedTrainSetFile = run_folder + "serializedTrain"
    serializedValSetFile = run_folder + "serializedVal"
    num_frames_per_video = 16
    imageFormat = "jpg"
    trainValRatio = 2.0 / 3.0
    data_format = None

    # input frames settings
    raw_image_shape = None
    image_shape = None
    input_mode = None

    # mean image subtraction
    mean_image = None
    do_mean_subtraction = None

    # image processing
    do_random_mirroring = None
    crop_w_avail = None
    crop_h_avail = None


    # per-class video indexes
    trainPerClass = []
    valPerClass = []

    # total list
    trainSet = []
    trainLabels = []
    valSet = []
    valLabels = []

    phase = None
    epoch_index = None
    epochs = None

    # current phase
    batch_count = 0
    batch_index = 0
    batch_size = None
    batches = None
    iterator = None

    # training
    batch_size_train = None
    batches_train = []
    batch_index_train = None
    train_iterator = None

    # validation
    do_validation = False
    validation_interval = None
    batch_size_val = None
    batches_val = []
    batch_index_val = None
    val_iterator = None


    # misc
    logger = None

    # mean subtraction
    def shoud_subtract_mean(self):
        return self.mean_image is not None
    # get class name from class index
    def getClassName(self,classIndex):
        return self.videoClassNames[classIndex]

    # read paths to video folders
    def read_video_metadata(self):
        self.logger.info("Reading video paths from %s" % self.videoFramePathsFile)
        with open(self.videoFramePathsFile) as f:
            for line in f:
                self.videoPaths.append(line.strip())
        with open(self.class_names) as f:
            for line in f:
                self.videoClassNames.append(line.strip())
        self.num_classes = len(self.videoClassNames)

    # read paths to images
    def read_frames_metadata(self):
        # store train and test in same variable
        self.logger.info("Reading input frames in single frame mode.");
        self.frame_paths.extend([[],[]])
        self.frame_classes.extend([[], []])
        with open(self.input_source_files[defs.phase.train], 'r') as f:
            for path_label_str in f:
                item_path, item_label = path_label_str.split()
                if self.path_prepend_folder is not None:
                    item_path = os.path.join(self.path_prepend_folder, item_path)
                self.frame_paths[defs.phase.train].append(item_path)
                self.frame_classes[defs.phase.train].append(item_label)

            with open(self.input_source_files[defs.phase.val], 'r') as f:
                for path_label_str in f:
                    item_path, item_label = path_label_str.split()
                    self.frame_paths[defs.phase.val].append(item_path)
                    self.frame_classes[defs.phase.val].append(item_label)

        self.logger.info("Done reading, %d frames for %s, %d frames for %s." % (
            len(self.frame_paths[defs.phase.train]), defs.phase.str(defs.phase.train),
            len(self.frame_paths[defs.phase.val]), defs.phase.str(defs.phase.val),
        ))

    # display image
    def display_image(self,image,label=None):
        # print(label)
        #plt.title(label)
        # simple casting
        # plt.imshow(np.ndarray.astype(image,np.uint8))
        # back to 0,255

        plt.imshow(np.ndarray.astype(image, np.uint8))
        plt.show()
        # plt.waitforbuttonpress()

    # read from tfrecord
    def deserialize_from_tfrecord(self, iterator, images_per_iteration):
        # images_per_iteration :
        images = []
        labels  = []
        for _ in range(images_per_iteration):
            try:
                string_record = next(iterator)
                example = tf.train.Example()
                example.ParseFromString(string_record)
                img_string = (example.features.feature['image_raw']
                              .bytes_list
                              .value[0])
                # height = int(example.features.feature['height']
                #              .int64_list
                #              .value[0])
                # width = int(example.features.feature['width']
                #             .int64_list
                #             .value[0])
                #
                # depth = (example.features.feature['depth']
                #          .int64_list
                #          .value[0])
                label = (example.features.feature['label']
                         .int64_list
                         .value[0])
                img_1d = np.fromstring(img_string, dtype=np.uint8)
                # watch it : hardcoding preferd dimensions according to the dataset object.
                # it should be the shape of the stored image instead, for generic use
                image = img_1d.reshape((self.raw_image_shape[0], self.raw_image_shape[1], self.raw_image_shape[2]))

                image = self.process_image(image)
                images.append(image)
                labels.append(label)

            except StopIteration:
                break
            except Exception as ex:
                self.logger.error('Exception at reading image, loading from scratch')
                self.logger.error(ex)
                error("Error reading tfrecord image.")

        return images, labels

    # set dataset phase
    def set_phase(self, phase):
        # if the network is switching phases, store the index
        # to continue later
        if self.phase == phase:
            return
        if self.phase is not None:
            self.logger.info("Suspending phase [%s], on current batch index [%d]" % (defs.phase.str(self.phase),
                                                                           self.batch_index))
            if self.phase == defs.phase.train:
                self.batch_index_train = self.batch_index
            else:
                self.batch_index_val = self.batch_index

        self.logger.info("Setting phase [%s], on batch index [%d]" % (defs.phase.str(phase),
                                                                        self.batch_index_val))
        # update phase variables
        self.phase = phase
        if self.phase == defs.phase.train:
            self.batches = self.batches_train
            self.batch_index = self.batch_index_train
            self.iterator = self.train_iterator
            self.batch_size = self.batch_size_train
        else:
            self.batches = self.batches_val
            self.batch_index = self.batch_index_val
            self.iterator = self.val_iterator
            self.batch_size = self.batch_size_val

    # read next batch
    def read_next_batch(self):
        images = []
        labels = []
        currentBatch = self.batches[self.batch_index]

        # switch read mode
        if self.data_format == defs.data_format.raw:
            # read stuff from disk directly
            if self.input_mode == defs.input_mode.video:
                # read video
                for videopath in currentBatch[defs.images]:
                    videoframes = self.get_video_frames(videopath)
                    images.extend(videoframes)
                    labels.append(currentBatch[defs.labels])
            else:
                # read image
                for impath in currentBatch[0]:
                    frame = self.read_image(impath, self.do_mean_subtraction)
                    images.append(frame)
                    labels = currentBatch[defs.labels]

        else:
            # read images from a TFrecord serialization file
            num_items_in_batch = currentBatch
            if self.input_mode == defs.input_mode.video:
                num_items_in_batch = num_items_in_batch * self.num_frames_per_video
            images, labels = self.deserialize_from_tfrecord(self.iterator, num_items_in_batch)
            # limit to 1 label per video
            if self.input_mode == defs.input_mode.video:
                labels = [ labels[l] for l in range(0,len(labels),self.num_frames_per_video) ]


        self.logger.debug("Fetching batch of %d images and %d labels." % (len(images), len(labels)))

        labels_onehot = labels_to_one_hot(labels, self.num_classes)
        self.batch_index = self.batch_index + 1
        return images, labels_onehot#, labels

    # read all frames for a video
    def get_video_frames(self,videopath):
        self.logger.debug("Reading frames of video idx %s from disk." % videopath)

        frames = []
        for im in range(self.num_frames_per_video):
            impath = "%s.%04d.%s" % (videopath, 1+im, self.imageFormat)
            frames.append(self.read_image(impath))
        return frames

    # get a random image crop
    def random_crop(self, image):
        randh = choice(self.crop_h_avail)
        randw = choice(self.crop_w_avail)
        image = image[
                randh:randh + self.image_shape[0],
                randw:randw + self.image_shape[1],
                :]
        return image

    # read image from disk
    def read_image(self,imagepath):
        image = imread(imagepath)

        self.logger.debug("Reading image %s" % imagepath)
        # for grayscale images, duplicate
        # intensity to color channels
        if len(image.shape) <= 2:
            image = np.repeat(image[:, :, np.newaxis], 3, 2)
        # drop channels other than RGB
        image = image[:,:,:3]
        #  convert to BGR
        image = image[:, :, ::-1]

        image = self.process_image(image)

        return image

    # apply image post-process
    def process_image(self, image):
        # take cropping
        if self.do_random_cropping:
            image = self.random_crop(image)
        else:
            image = imresize(image, self.raw_image_shape)

        if self.do_mean_subtraction:
            image = image - self.mean_image

        if self.do_random_mirroring:
            if not randrange(2):
                image = image[:,::-1,:]
        return image

    # do preparatory work
    def initialize(self, sett):
        # transfer configuration from settings
        self.logger = sett.logger
        self.epochs = sett.epochs
        self.do_random_mirroring = sett.do_random_mirroring
        self.do_validation = sett.do_validation
        self.validation_interval = sett.validation_interval
        self.run_folder = sett.runFolder
        self.path_prepend_folder = sett.path_prepend_folder

        self.data_format = sett.frame_format
        self.input_mode = sett.input_mode
        self.mean_image = sett.mean_image
        self.raw_image_shape = sett.raw_image_shape
        self.num_classes = sett.num_classes
        self.do_random_cropping = sett.do_random_cropping
        self.image_shape = sett.image_shape
        self.logger.info("Initializing run on folder [%s]" % self.run_folder)

        self.batch_size_train = sett.batch_size_train
        self.batch_size_val = sett.batch_size_val
        self.initialize_data(sett)

        # transfer resumed snapshot settings
        self.epoch_index = sett.epoch_index
        self.batch_index_train = sett.train_index
        self.batch_index_val = sett.val_index


        self.do_mean_subtraction = (self.mean_image is not None)
        if self.do_mean_subtraction:
            # build the mean image
            height = self.image_shape[0]
            width = self.image_shape[1]
            blue = np.full((height, width), self.mean_image[0])
            green = np.full((height, width), self.mean_image[1])
            red = np.full((height, width), self.mean_image[2])
            self.mean_image = np.ndarray.astype(np.stack([blue, green, red]),np.float32)
            self.mean_image = np.transpose(self.mean_image, [1, 2, 0])

        if self.do_random_cropping:
            self.crop_h_avail = [i for i in range(0, self.raw_image_shape[0] - self.image_shape[0] - 1)]
            self.crop_w_avail = [i for i in range(0, self.raw_image_shape[1] - self.image_shape[1] - 1)]

        self.logger.info("Completed dataset initialization.")
        self.tell()

    # run data-related initialization pre-run
    def initialize_data(self, sett):
        self.logger.info("Initializing %s data on input mode %s." % (defs.data_format.str(self.data_format), defs.input_mode.str(self.input_mode)))
        if self.data_format == defs.data_format.tfrecord:
            # initialize tfrecord, check file consistency
            self.input_source_files[defs.phase.train] = sett.input[defs.phase.train] + ".tfrecord"
            self.input_source_files[defs.phase.val] = sett.input[defs.phase.val] + ".tfrecord"

            if not os.path.exists(self.input_source_files[defs.phase.train]):
                error("Input file does not exist: %s" % self.input_source_files[defs.phase.train])
            if not os.path.exists(self.input_source_files[defs.phase.train]):
                error("Input file does not exist: %s" % self.input_source_files[defs.phase.train])

            self.reset_iterator(defs.phase.train)
            self.reset_iterator(defs.phase.val)
            # count or get number of items
            num_train = self.get_input_data_count(defs.phase.train)
            num_val = self.get_input_data_count(defs.phase.val)

            # restore batch index to snapshot if needed
            if self.batch_index > 0:
                self.fast_forward_iter(self.iterator)

            # calculate batches: just batchsizes per batch; all is in the tfrecord
            num_whole_batches = num_train // self.batch_size_train
            self.batches_train = [ self.batch_size_train for i in range(num_whole_batches)]
            items_left = num_train - num_whole_batches * self.batch_size_train
            if items_left:
                self.batches_train.append(items_left)

            num_whole_batches = num_val // self.batch_size_val
            self.batches_val = [ self.batch_size_train for i in range(num_whole_batches)]
            items_left = num_val - num_whole_batches * self.batch_size_val
            if items_left:
                self.batches_val.append(items_left)


        else:
            # read input files
            self.input_source_files[defs.phase.train] = sett.input[defs.phase.train]
            self.input_source_files[defs.phase.val] = sett.input[defs.phase.val]
            # read frames and classes
            self.read_frames_metadata()

            # for raw input mode, calculate batches for paths and labels

            # train
            imgs = sublist(self.frame_paths[defs.phase.train], self.batch_size_train)
            lbls = sublist(self.frame_classes[defs.phase.train], self.batch_size_train)
            for l in range(len(lbls)):
                self.batches_train.append([
                    imgs[l],
                    list(map(int, lbls[l]))
                         ])
            # val
            imgs = sublist(self.frame_paths[defs.phase.val], self.batch_size_val)
            lbls = sublist(self.frame_classes[defs.phase.val], self.batch_size_val)
            for l in range(len(lbls)):
                self.batches_val.append([
                    imgs[l],
                    list(map(int, lbls[l]))
                         ])

    # read or count how many data items are in training / validation
    def get_input_data_count(self,phase):
        if phase == defs.phase.train:
            input_file  = self.input_source_files[defs.phase.train]
            iterator = self.train_iterator
        else:
            input_file = self.input_source_files[defs.phase.val]

        if not os.path.exists(input_file + ".size"):
            num = 0
            num_print = 1000
            self.logger.info("No size file for %s data. Counting..." % defs.phase.str(phase))
            try:
                for _ in iterator:
                    num = num + 1
                    if num > 0 and num % num_print == 0:
                        self.logger.info("Counted %d instances so far." % num)
            except StopIteration:
                pass
            self.logger.info("Counted tfrecord %s data: %d entries." % (defs.phase.str(phase), num))
            self.reset_iterator(iterator)
        else:
            with open(input_file + ".size","r") as ff:
                num = int(ff.read())
            self.logger.info("Got tfrecord %s data: %d entries." % (defs.phase.str(phase), num))

        return num

    # set iterator to point to the beginning of the tfrecord file, per phase
    def reset_iterator(self,phase):
        if not self.data_format == defs.data_format.tfrecord:
            return
        if phase == defs.phase.train:
            self.train_iterator = tf.python_io.tf_record_iterator(path=self.input_source_files[defs.phase.train])
        else:
            self.val_iterator = tf.python_io.tf_record_iterator(path=self.input_source_files[defs.phase.val])

    # move up an iterator
    def fast_forward_iter(self):

        # fast forward to batch index
        num_forward = self.batch_index
        if self.input_mode == defs.input_mode.image:
            self.logger.info("Fast forwarding to the batch # %d ( image # %d)" % (self.batch_index+1, num_forward+1))
        elif self.input_mode == defs.input_mode.video:
            num_forward = num_forward * self.num_frames_per_video
            self.logger.info("Fast forwarding to the batch # %d ( image # %d )" % (self.batch_index + 1, num_forward + 1))


        for _ in range(num_forward):
            next(self.iterator)

    # load saved parameters
    def load_metaparams(self, params):
        i = 0
        self.batch_index_train = params[i]; i+=1
        self.epoch_index = params[i]; i+=1
        self.set_file_paths()

    # print active settings
    def tell(self):
        self.logger.info("Dataset batch information per run mode:" )
        self.logger.info("%-8s %-6s %-6s %-6s" % ("Mode","size","num","index"))
        self.logger.info("%-8s %-6d %-6d %-6d" % (defs.phase.str(defs.phase.train), self.batch_size_train, len(self.batches_train), self.batch_index_train))
        self.logger.info("%-8s %-6d %-6d %-6d" % (defs.phase.str(defs.phase.val), self.batch_size_val, len(self.batches_val), self.batch_index_val))

    # get the batch size
    def get_batch_size(self):
        if self.phase == "train":
            return self.batch_size_train
        else:
            return self.batch_size_val

    # get the global step
    def get_global_step(self):
        return self.epoch_index * self.batch_size_train + self.batch_index_train

    # print iteration information
    def print_iter_info(self, num_images, num_labels):
        self.logger.info("Mode: [%s], epoch: %2d/%2d, batch %4d / %4d : %3d images, %3d labels" %
                         (defs.phase.str(self.phase), self.epoch_index + 1, self.epochs, self.batch_index, len(self.batches), num_images, num_labels))

    # specify valid loop iteration
    def loop(self):
        return self.batch_index < len(self.batches)

    # reset data reading params for phase
    def reset_phase(self, phase):
        if phase == defs.phase.train:
            self.batch_index_train = 0
            self.reset_iterator(defs.phase.train)
        else:
            self.batch_index_val = 0
            self.reset_iterator(defs.phase.val)
        # pass on the values to the current phase
        self.phase = None
        self.set_phase(phase)

    # check if testing should happen now
    def should_test_now(self):
        return self.do_validation and (self.batch_count > 0) and ((self.batch_count // self.validation_interval) == 0)
