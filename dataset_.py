
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

    # single frame paths and classes, for raw input mode
    frame_paths = []
    frame_classes = []


    # run settings
    run_folder = None
    clips_per_video = None
    num_frames_per_clip = None
    frame_format = None
    data_format = None

    # input frames settings
    raw_image_shape = None
    image_shape = None
    input_mode = None

    # mean image subtraction
    mean_image = None
    do_mean_subtraction = None

    # frame processing
    do_random_mirroring = None
    do_random_cropping = None
    crop_w_avail = None
    crop_h_avail = None

    # epoch and phase variables
    phase = None
    epoch_index = None
    epochs = None

    # current phase variables
    batch_count = 0
    batch_index = 0
    batch_size = None
    batches = None
    iterator = None

    # training variables
    do_training = None
    batch_size_train = None
    batches_train = []
    batch_index_train = None
    train_iterator = None
    num_items_train = 0

    # validation variables
    do_validation = False
    validation_interval = None
    batch_size_val = None
    batches_val = []
    batch_index_val = None
    val_iterator = None
    num_items_val = 0

    # misc
    logger = None

    # mean subtraction
    def shoud_subtract_mean(self):
        return self.mean_image is not None

    # read paths to images
    def read_frames_metadata(self):
        # store train and test in same variable
        self.logger.info("Reading input frames in raw frame mode.");
        self.frame_paths.extend([[],[]])
        self.frame_classes.extend([[], []])

        if self.do_training:
            with open(self.input_source_files[defs.phase.train], 'r') as f:
                for path_label_str in f:
                    item_path, item_label = path_label_str.split()
                    if self.path_prepend_folder is not None:
                        item_path = os.path.join(self.path_prepend_folder, item_path)
                    self.frame_paths[defs.phase.train].append(item_path)
                    self.frame_classes[defs.phase.train].append(item_label)
            self.logger.info("Done reading image and label data %d frames for %s" % (
                len(self.frame_paths[defs.phase.train]), defs.phase.str(defs.phase.train),
            ))

        if self.do_validation:
            with open(self.input_source_files[defs.phase.val], 'r') as f:
                for path_label_str in f:
                    item_path, item_label = path_label_str.split()
                    self.frame_paths[defs.phase.val].append(item_path)
                    self.frame_classes[defs.phase.val].append(item_label)

            self.logger.info("Done reading image and label data %d frames for %s" % (
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

    # set dataset phase, but keep track of potential current
    def set_or_swap_phase(self, phase):
        # if the network is switching phases, store the index
        # to continue later
        if self.phase == phase:
            return
        if self.phase is not None:
            self.logger.info("Suspending phase [%s] @ batch [%d]" % (defs.phase.str(self.phase), self.batch_index))
            if self.phase == defs.phase.train:
                self.batch_index_train = self.batch_index
            else:
                self.batch_index_val = self.batch_index
        self.set_current_phase(phase)

    # assign the current phase variables to the argument phase
    def set_current_phase(self, phase, is_new_phase = True):

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
        if is_new_phase:
            self.logger.info("Starting phase [%s] @ batch [%d]." % (defs.phase.str(phase), self.batch_index))

    # reset data reading params for phase
    def reset_phase(self, phase):
        self.logger.info("Reseting phase [%s]." % defs.phase.str(phase))
        if phase == defs.phase.train:
            self.batch_index_train = 0
            self.reset_iterator(defs.phase.train)
        else:
            self.batch_index_val = 0
            self.reset_iterator(defs.phase.val)
        # if argument phase is the current one, also update current phase variables
        if phase == self.phase:
            self.set_current_phase(phase, is_new_phase=False)


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
                    frame = self.read_image(impath)
                    images.append(frame)
                    labels = currentBatch[defs.labels]

        else:
            # read images from a TFrecord serialization file
            num_items_in_batch = currentBatch
            if self.input_mode == defs.input_mode.video:
                num_items_in_batch = num_items_in_batch * self.num_frames_per_clip * self.clips_per_video
            images, labels = self.deserialize_from_tfrecord(self.iterator, num_items_in_batch)
            # limit to 1 label per video
            if self.input_mode == defs.input_mode.video:
                labels = [ labels[l] for l in range(0,len(labels),self.num_frames_per_clip) ]

        labels_onehot = labels_to_one_hot(labels, self.num_classes)
        self.advance_batch_index()
        return images, labels_onehot#, labels

    def advance_batch_index(self):
        self.batch_index = self.batch_index + 1
        if self.phase == defs.phase.train:
            self.batch_index_train = self.batch_index_train + 1
        else:
            self.batch_index_val = self.batch_index_val + 1

    # read all frames for a video
    def get_video_frames(self,videopath):
        self.logger.debug("Reading frames of video idx %s from disk." % videopath)

        frames = []
        for im in range(self.num_frames_per_clip):
            impath = "%s%04d.%s" % (videopath, 1+im, self.frame_format)
            frame = self.read_image(impath)
            frames.append(frame)
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
            image = imresize(image, self.image_shape)

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
        self.do_training = sett.do_training
        self.do_validation = sett.do_validation
        self.validation_interval = sett.validation_interval
        self.run_folder = sett.run_folder
        self.path_prepend_folder = sett.path_prepend_folder

        self.data_format = sett.data_format
        self.frame_format = sett.frame_format
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

        self.logger.debug("Completed dataset initialization.")
        self.tell()

    # run data-related initialization pre-run
    def initialize_data(self, sett):
        self.logger.info("Initializing %s data on input mode %s." % (defs.data_format.str(self.data_format), defs.input_mode.str(self.input_mode)))
        if self.data_format == defs.data_format.tfrecord:

            # initialize tfrecord, check file consistency
            if self.do_training:
                self.input_source_files[defs.phase.train] = sett.input[defs.phase.train] + ".tfrecord"
                if not os.path.exists(self.input_source_files[defs.phase.train]):
                    error("Input file does not exist: %s" % self.input_source_files[defs.phase.train])
                self.reset_iterator(defs.phase.train)
                # count or get number of items
                self.get_input_data_count(defs.phase.train)
                # restore batch index to snapshot if needed - only makes sense in training
                if self.batch_index > 0:
                    self.fast_forward_iter(self.train_iterator)
                # calculate batches: just batchsizes per batch; all is in the tfrecord
                num_whole_batches =  self.num_items_train // self.batch_size_train
                self.batches_train = [self.batch_size_train for _ in range(num_whole_batches)]
                items_left =  self.num_items_train - num_whole_batches * self.batch_size_train
                if items_left:
                    self.batches_train.append(items_left)
                self.logger.info("Calculated %d training batches." % len(self.batches_train))

            if self.do_validation:
                self.input_source_files[defs.phase.val] = sett.input[defs.phase.val] + ".tfrecord"
                if not os.path.exists(self.input_source_files[defs.phase.val]):
                    error("Input file does not exist: %s" % self.input_source_files[defs.phase.val])
                self.reset_iterator(defs.phase.val)
                # count or get number of items
                self.get_input_data_count(defs.phase.val)
                num_whole_batches = self.num_items_val // self.batch_size_val
                self.batches_val = [ self.batch_size_val for i in range(num_whole_batches)]
                items_left = self.num_items_val - num_whole_batches * self.batch_size_val
                if items_left:
                    self.batches_val.append(items_left)
                self.logger.info("Calculated %d validation batches." % len(self.batches_val))
        elif self.data_format == defs.data_format.raw:
            # read input files
            self.input_source_files[defs.phase.train] = sett.input[defs.phase.train]
            self.input_source_files[defs.phase.val] = sett.input[defs.phase.val]

            # read frames and classes
            self.read_frames_metadata()
            if self.do_training:

                # for raw input mode, calculate batches for paths and labels
                imgs = sublist(self.frame_paths[defs.phase.train], self.batch_size_train)
                lbls = sublist(self.frame_classes[defs.phase.train], self.batch_size_train)
                for l in range(len(lbls)):
                    self.batches_train.append([
                        imgs[l],
                        list(map(int, lbls[l]))
                    ])
            if self.do_validation:
                imgs = sublist(self.frame_paths[defs.phase.val], self.batch_size_val)
                lbls = sublist(self.frame_classes[defs.phase.val], self.batch_size_val)
                for l in range(len(lbls)):
                    self.batches_val.append([
                        imgs[l],
                        list(map(int, lbls[l]))
                    ])
            self.num_items_train = len(self.frame_paths[defs.phase.train])
            self.num_items_val = len(self.frame_paths[defs.phase.val])

            self.num_items_train = len(self.frame_paths[defs.phase.train])
            self.num_items_val = len(self.frame_paths[defs.phase.val])
        else:
            self.logger.error("Undefined data format [%s]" % (defs.data_format.str(self.data_format)))
            error("Data format error")


    # read or count how many data items are in training / validation
    def get_input_data_count(self,phase):
        # init reading
        if phase == defs.phase.train:
            input_file  = self.input_source_files[defs.phase.train]
            iterator = self.train_iterator
        else:
            input_file = self.input_source_files[defs.phase.val]
            iterator = self.val_iterator

        # check if a .size file exists
        if not os.path.exists(input_file + ".size"):
            # does not. Just count the entries :(
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
            if phase == defs.phase.train:
                self.num_items_train = num
            elif phase == defs.phase.val:
                self.num_items_val = num
            return


        # .size file  exists. Read the count and optionally # frames per clip and # clips
        # format expected is one value per line, in the order of num_items, num_frames_per_clip, num_clips_per_video
        # read the data
        size_file = input_file + ".size"
        self.logger.info("Reading data meta-parameters file [%s]" % size_file)
        contents = read_file_lines(size_file)

        # check contents
        if len(contents) == 0 :
            # no number of frames per video, it's gotta be an image run, else error
            if not self.input_mode == defs.input_mode.image:
                self.logger.error(
                    "Specified input mode %s but size file contains no data" %
                    (defs.input_mode.str(self.input_mode)))
                error("Input mode mismatch with data.")

        # read number of items in the tfrecord
        num_items = int(contents[0])
        if phase == defs.phase.train:
            self.num_items_train = num_items
        elif phase == defs.phase.val:
            self.num_items_val = num_items
        self.logger.info("Read a count of %d [%s] items in the input data" % ( phase, num_items))



        # read number of frames per clip
        if len(contents) > 1:
            num_frames_per_clip = int(contents[1])
            # check if there's a clash with the run configuration specified
            if self.num_frames_per_clip is not None:
                if not num_frames_per_clip == self.num_frames_per_clip:
                    self.logger.error("Read %d frames per clip from the size file but specified %d" %
                                      (num_frames_per_clip,  self.num_frames_per_clip))
                    error("Number of video frames mismatch")
            else:
                self.logger.info("Read %d frames per clip for %s from size file " %
                                  (num_frames_per_clip, defs.phase.str(phase)))
            self.num_frames_per_clip = num_frames_per_clip
        else:
            if self.input_mode == defs.input_mode.video:
                self.logger.error("Specified %s input mode but no frames per clip information in input data." % defs.input_mode.str(self.input_mode))
                error("Missing frames per clip information in data")

        # read number of clips per video
        if len(contents) > 2 :
            try:
                # read a single number of clips, for each video
                num_clips = int(contents[2])
                if self.clips_per_video is not None:
                    self.logger.error(
                        "Read value of clips per video clashes with already set value of %d " % self.clips_per_video)
                else:
                    self.logger.info("Read a value of %d clips per video " % num_clips)
                self.clips_per_video = num_clips
            except Exception:
                # read a collection of numbers, each denoting the number of clips per video
                vals = contents[2].split(" ")
                num_clips = []
                for val in vals:
                    val = val.strip()
                    num_clips.append(int(val))
                self.logger.info ("Read %d values of number of clips per video" % (len(num_clips)))

        else:
            # else, if unset, set the default number of clips to 1
            self.logger.warning("No number of clips in size file, defaulting to 1 clip per video.")
            self.clips_per_video = 1

        # if a single value was read for cpv, expand to number of videos
        if type(self.clips_per_video) == list:
            if not len(self.clips_per_video) == num_items:
                self.logger.error("Unequal number of clips vector %d to the number of videos %d" % (len(self.clips_per_video),num_items))
                error("Unequal number of clips vector to the input videos.")
        else:
            # replicate the single value for each video
            self.clips_per_video = [ self.clips_per_video for _ in range(num_items) ]

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
            num_forward = num_forward * self.num_frames_per_clip
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
        self.logger.info("%-8s %-10s %-6s %-6s %-6s" % ("Mode","total","b-size","num-b","b-index"))
        self.logger.info("%-8s %-10d %-6d %-6d %-6d" % (defs.phase.str(defs.phase.train), self.num_items_train, self.batch_size_train, len(self.batches_train), self.batch_index_train))
        self.logger.info("%-8s %-10d %-6d %-6d %-6d" % (defs.phase.str(defs.phase.val), self.num_items_val, self.batch_size_val, len(self.batches_val), self.batch_index_val))

    # get the batch size
    def get_batch_size(self):
        if self.phase == defs.phase.train:
            return self.batch_size_train
        else:
            return self.batch_size_val

    # get the global step
    def get_global_step(self):
        return self.epoch_index * self.batch_size_train + self.batch_index_train

    # print iteration information
    def print_iter_info(self, num_images, num_labels):
        if self.phase == defs.phase.train:
            self.logger.info("Mode: [%s], epoch: %2d/%2d, batch %4d / %4d : %3d images, %3d labels" %
                         (defs.phase.str(self.phase), self.epoch_index + 1, self.epochs, self.batch_index, len(self.batches), num_images, num_labels))
        # same as train, but no epoch
        elif self.phase == defs.phase.val:
            self.logger.info("Mode: [%s], batch %4d / %4d : %3d images, %3d labels" %
                         (defs.phase.str(self.phase), self.batch_index, len(self.batches), num_images, num_labels))

    # specify valid loop iteration
    def loop(self):
        self.logger.debug("Checking loop @ batch index: %d , batches len: %d" % (self.batch_index+1 , len(self.batches)))
        return self.batch_index < len(self.batches)

    # check if testing should happen now
    def should_test_now(self):
        retval = self.do_validation and (self.batch_count > 0) and ((self.batch_count % self.validation_interval) == 0)
        self.batch_count = self.batch_count + 1
        return retval


