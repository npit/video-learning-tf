
# files and io
import os, pickle, math

# utils
from utils_ import *
from random import choice, randrange
# image IO
from scipy.misc import imread, imresize, imsave
from defs_ import *
# displaying
#import matplotlib.pyplot as plt

# Dataset class
class Dataset:
    workflow = None
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
    batch_item = None

    # variables for validation clip batches
    video_index = 0

    # misc
    embedding_matrix = None
    vocabulary = None

    save_interval = None

    # mean subtraction
    def shoud_subtract_mean(self):
        return self.mean_image is not None

    # read paths to images
    def read_frames_metadata(self):
        # store train and test in same variable
        self.frame_paths.extend([[],[]])
        self.frame_classes.extend([[], []])

        if self.do_training:
            with open(self.input_source_files[defs.train_idx], 'r') as f:
                for path_label_str in f:
                    path_labels = path_label_str.split()
                    item_path = path_labels[0]
                    item_labels = path_labels[1:]
                    if self.path_prepend_folder is not None:
                        item_path = os.path.join(self.path_prepend_folder, item_path)
                    self.frame_paths[defs.train_idx].append(item_path)
                    self.frame_classes[defs.train_idx].append(item_labels)

        if self.do_validation:
            with open(self.input_source_files[defs.val_idx], 'r') as f:
                for path_label_str in f:
                    path_labels = path_label_str.split()
                    item_path = path_labels[0]
                    item_labels = path_labels[1:]
                    self.frame_paths[defs.val_idx].append(item_path)
                    self.frame_classes[defs.val_idx].append(item_labels)

    # display image
    def display_image(self,image,label=None):
        # print(label)
        #plt.title(label)
        # simple casting
        # plt.imshow(np.ndarray.astype(image,np.uint8))
        # back to 0,255

        # plt.imshow(np.ndarray.astype(image, np.uint8))
        # plt.show()
        # plt.waitforbuttonpress()
        pass

    # read from tfrecord
    def deserialize_from_tfrecord(self, iterator, images_per_iteration):
        # images_per_iteration :
        images = []
        labels  = []
        for imidx in range(images_per_iteration):
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
                         .value)
                label = list(label)
                label = label[0] if len(label) == 0 else label


                img_1d = np.fromstring(img_string, dtype=np.uint8)
                # watch it : hardcoding preferd dimensions according to the dataset object.
                # it should be the shape of the stored image instead, for generic use
                image = img_1d.reshape((self.raw_image_shape[0], self.raw_image_shape[1], self.raw_image_shape[2]))




            except StopIteration:
                if imidx < images_per_iteration:
                    warning('Tfrecord unexpected EOF, loading from scratch')
                    image, label = self.manually_read_image(imidx)


            except Exception as ex:
                warning('Exception at reading image, loading from scratch')
                image,label = self.manually_read_image(imidx)

            image = self.process_image(image)
            images.append(image)
            labels.append(label)


        return images, labels

    # fallback when tfrecord image reads fail
    def manually_read_image(self, idx_in_batch):
        global_image_index = self.batch_index * self.batch_size + idx_in_batch
        phase = defs.train_idx if self.phase == defs.phase.train else defs.val_idx
        impath = self.frame_paths[phase][global_image_index]
        info("Attempting to manually read global image index %d : %s" % (global_image_index, impath))
        image = self.read_image(impath)
        image = imresize(image, self.raw_image_shape)
        label = self.frame_classes[phase][global_image_index]
        label = list(map(int,label))
        return image,label

    # set dataset phase, but keep track of potential current
    def set_or_swap_phase(self, phase):
        # if phase already set, done
        if self.phase == phase:
            return
        # if the network is switching phases, store the index
        # to continue later
        if self.phase is not None:
            info("Suspending phase [%s] @ batch [%d]" % (self.phase, self.batch_index))
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
            info("Starting phase [%s] @ batch [%d]." % (phase, self.batch_index))

    # reset data reading params for phase
    def reset_phase(self, phase):
        info("Reseting phase [%s]." % phase)
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
    def get_next_batch(self):
        images = []
        labels = []
        currentBatch = self.batches[self.batch_index]

        # switch read mode
        if self.data_format == defs.data_format.raw:
            # read stuff from disk directly
            if self.input_mode == defs.input_mode.video:
                # read video
                for videopath in currentBatch[0]:
                    videoframes = self.get_video_frames(videopath)
                    images.extend(videoframes)
                    labels.append(currentBatch[1])
            else:
                # read image
                for impath in currentBatch[0]:
                    frame = self.read_image(impath)
                    images.append(frame)
                    labels = currentBatch[1]
        else:
            # for video mode, get actual number of frames for the batch items
            if self.input_mode == defs.input_mode.video:
                images, labels = self.get_next_batch_video_tfr()
            else:
                images, labels = self.get_next_batch_frame_tfr(currentBatch)

        if defs.workflows.is_description(self.workflow):

            if self.do_validation:
                # pad incomplete batch with zeros in both images and captions
                num_pad = self.batch_size_val - len(images)
                images.extend( [ np.zeros([1, images[0].shape[1]],images[0].dtype) for _ in range(num_pad)] )
                labels.extend( [[-1] for _ in range(num_pad)])
            ground_truth = self.labels_to_words(labels)
        else:
            ground_truth = labels_to_one_hot(labels, self.num_classes)

        self.advance_batch_indexes()
        return images, ground_truth



    # get captions from logit vectors
    def logits_to_captions(self, logits):
        return_data = []
        captions = []
        # for validation mode, we better have the image ids
        if self.phase == defs.phase.val:
            if self.eval_type == defs.eval_type.coco:
                image_ids = []
                parts = self.input_source_files[defs.val_idx].split(".")
                image_paths_file = ".".join(parts[0:-1])
                info("Reading image paths for id extraction from %s" % image_paths_file)
                # read image ids, presumed to be the suffixless basename
                with open(image_paths_file, "r") as fp:
                    for line in fp:
                        # format example : COCO_val2014_000000000042.jpg
                        # strip id from path
                        try:
                            # get filename
                            parts = line.strip().split()
                            filename = os.path.basename(parts[0]).split('.')[0]
                            # get image id
                            image_id = filename.split("_") [-1]
                            image_id = int(image_id)
                        except Exception:
                            warning("Could not convert image id %s to int. Storing as string." % image_id)
                        image_ids.append(image_id)
                # return it
                return_data.append(image_ids)
        # read captions
        for i,image_logits in enumerate(logits):
            image_caption = []
            for caption_index in image_logits:
                image_caption.append(self.vocabulary[caption_index])
            if not image_caption:
                image_caption = ' '
            captions.append(" ".join(image_caption))
        return_data.append(captions)
        debug("Generated ids:")
        for i in range(len(image_ids)):
            debug("image id: %s caption:%s" % (str(image_ids[i]),str(captions[i])))
        return return_data

    # get word embedding vectors from vocabulary encoded indices
    def labels_to_words(self, batch_labels):
        # initialize to empties
        batch_word_vectors = np.zeros([0, self.embedding_matrix.shape[1]], np.float32)
        batch_onehot_labels = np.zeros([0, self.num_classes], np.int32)
        batch_no_pad_index = []
        embedding_dim = self.embedding_matrix.shape[1]
        # get special tokens indices
        bos_index = self.vocabulary.index("BOS")
        eos_index = self.vocabulary.index("EOS")

        for batch_index, item_labels in enumerate(batch_labels):
            # debug print the labels
            input_labels = [bos_index] + item_labels
            output_labels = item_labels + [eos_index]
            debug("Item io labels:  %s , %s" % (str(input_labels), str(output_labels)))
            # get caption word embedding vector
            # set the BOS as the first input token
            image_word_vectors = self.embedding_matrix[bos_index, :]

            # if training, put the caption word vectors after the BOS in the input, else leave just BOS
            if self.do_training:
                image_word_vectors = np.vstack((image_word_vectors, self.embedding_matrix[item_labels, :]))
            # get labels needed for training loss / validation accuracy
            image_onehot_labels = np.vstack((labels_to_one_hot(item_labels, self.num_classes)))
            # append output labels with EOS, if in training mode
            if self.do_training:
                image_onehot_labels = np.vstack((image_onehot_labels, labels_to_one_hot([eos_index], self.num_classes)))

            # if the number of items is less than the batch size, pad with zeros
            # pad to the max number of words with zero vectors. Also keep track of non-pad entries
            num_input_labels = len(input_labels)
            num_left_to_max = self.max_sequence_length - num_input_labels
            local_indexes, offset = list(range(num_input_labels)), batch_index * (self.max_sequence_length)
            batch_no_pad_index.extend([offset + elem for elem in  local_indexes])
            if num_left_to_max > 0:
                image_word_vectors = np.vstack((image_word_vectors, np.zeros([num_left_to_max, embedding_dim], np.float32)))

            # append labels and word vectors to batch containers
            batch_onehot_labels = np.vstack((batch_onehot_labels, image_onehot_labels))
            batch_word_vectors = np.vstack((batch_word_vectors, image_word_vectors))


        # word vectors batch always contains max_caption_len x batch-size elements, due to padding.
        # batch labels contain the exact labels for each word for each image caption. I.e., no padding there.
        ground_truth = {}
        ground_truth['word_embeddings'] = batch_word_vectors
        ground_truth['onehot_labels'] = batch_onehot_labels
        ground_truth['caption_lengths'] = list(map(len, batch_labels))
        ground_truth['non_padding_index'] = batch_no_pad_index
        debug("Non-padding index in batch: %s" % str(batch_no_pad_index))
        return ground_truth

    def get_next_batch_video_tfr(self):
        if self.batch_item == defs.batch_item.default:
            # batch size refers to videos. Get num clips per videos in batch
            curr_video = self.batch_index * self.batch_size
            curr_cpv = self.clips_per_video[curr_video : curr_video + self.batch_size]
            num_frames_in_batch = sum([self.num_frames_per_clip * x for x in curr_cpv])
            # read the frames from the TFrecord serialization file
            images, labels_per_frame = self.deserialize_from_tfrecord(self.iterator, num_frames_in_batch)

            # limit to <numclips> labels per video.
            fpv = [self.num_frames_per_clip * clip for clip in curr_cpv]
            fpv = np.cumsum(fpv)
            first_videoframe_idx = [0, *fpv[:-1] ]
            labels = []
            for vidx in range(len(curr_cpv)):
                fvi = first_videoframe_idx [vidx]
                for _ in range(curr_cpv[vidx]):
                    labels.append(labels_per_frame[fvi])

        elif self.batch_item == defs.batch_item.clip:
            # batch size refers to number of clips.
            num_frames_in_batch = self.batch_size * self.num_frames_per_clip
            # handle potentially incomplete batches
            clips_left = len(self.batches) - self.batch_index
            num_frames_in_batch = math.min(clips_left * self.num_frames_per_clip, num_frames_in_batch)
            images, labels_per_frame = self.deserialize_from_tfrecord(self.iterator, num_frames_in_batch)
            # limit to 1 label per clip
            labels = labels_per_frame[0:: self.num_frames_per_clip]

        return images,labels

    def get_next_batch_frame_tfr(self, num_frames_in_batch):
        # read images from a TFrecord serialization file
        images, labels = self.deserialize_from_tfrecord(self.iterator, num_frames_in_batch)
        if self.input_mode == defs.input_mode.video:
            labels = [labels[l] for l in range(0, len(labels), self.num_frames_per_clip)]
        return images,labels

    def advance_batch_indexes(self):
        self.batch_index = self.batch_index + 1
        if self.phase == defs.phase.train:
            self.batch_index_train = self.batch_index_train + 1
        else:
            self.batch_index_val = self.batch_index_val + 1

    def get_num_items(self, *varargs):
        if len(varargs) > 0:
            phase = varargs[0]
        else:
            phase = self.phase
        if phase == defs.phase.train:
            return self.num_items_train
        elif phase == defs.phase.val:
            return self.num_items_val

    # read all frames for a video
    def get_video_frames(self,videopath):
        debug("Reading frames of video idx %s from disk." % videopath)

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

        debug("Reading image %s" % imagepath)
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
        # transfer relevant configuration from settings
        self.workflow = sett.workflow
        self.epochs = sett.epochs
        self.do_random_mirroring = sett.do_random_mirroring
        self.do_training = sett.do_training
        self.do_validation = sett.do_validation
        self.validation_interval = sett.validation_interval
        self.run_folder = sett.run_folder
        self.path_prepend_folder = sett.path_prepend_folder
        self.frame_encoding_layer = sett.frame_encoding_layer

        self.data_format = sett.data_format
        self.frame_format = sett.frame_format
        self.mean_image = sett.mean_image
        self.raw_image_shape = sett.raw_image_shape
        self.num_classes = sett.num_classes
        self.do_random_cropping = sett.do_random_cropping
        self.image_shape = sett.image_shape
        info("Initializing run on folder [%s]" % self.run_folder)

        self.batch_size_train = sett.batch_size_train
        self.batch_size_val = sett.batch_size_val
        self.batch_item = sett.batch_item
        self.batch_index_val = 0

        self.input_mode = defs.input_mode.get_from_workflow(self.workflow)

        # transfer resumed snapshot settings
        self.epoch_index = sett.epoch_index
        self.batch_index_train = sett.train_index

        self.caption_search = sett.caption_search
        self.eval_type = sett.eval_type

        # iniitalize image data
        self.initialize_data(sett)

        # perform run mode - specific initializations
        self.initialize_workflow(sett)

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

        # set save interval
        if (sett.save_freq_per_epoch is not None) and (sett.do_training):
            self.save_interval = math.ceil(len(self.batches_train) / sett.save_freq_per_epoch)
            info("Computed batch save interval (from %f per %d-batched epoch) to %d batches" %
                             (sett.save_freq_per_epoch, len(self.batches_train), self.save_interval))
        self.tell()

    # initialize run mode specific data
    def initialize_workflow(self, settings):
        if defs.workflows.is_description(self.workflow):
            # read embedding matrix
            info("Reading embedding matrix from file [%s]" % settings.word_embeddings_file)
            self.vocabulary = []
            vectors = []
            with open(settings.word_embeddings_file,"r") as f:
                for line in f:
                    token, vector = line.strip().split("\t")
                    # add to vocabulary checking for duplicates
                    if token in self.vocabulary:
                        error("Duplicate token [%s] in vocabulary!")
                    self.vocabulary.append(token)
                    vector = [float(v) for v in vector.split()]
                    vectors.append(vector)

            self.embedding_matrix = np.asarray(vectors, np.float32)
            debug("Read embedding matrix of shape %s " % str(self.embedding_matrix.shape))
            if "BOS" not in self.vocabulary:
                error("BOS not found in vocabulary.")
            if "EOS" not in self.vocabulary:
                error("EOS not found in vocabulary.")
            if self.vocabulary.index("BOS") != len(self.vocabulary) -1:
                error("The BOS index in the vocabulary has be the last one (%d), but it currently is %d" \
                       % (len(self.vocabulary) - 1, self.vocabulary.index("BOS")) )
            # classes are all tokens minus the BOS
            self.num_classes = len(self.vocabulary) - 1

    def get_embedding_dim(self):
        return int(self.embedding_matrix.shape[-1])


    # run data-related initialization pre-run
    def initialize_data(self, sett):
        info("Initializing %s data on input mode %s." % (self.data_format, self.input_mode))
        if self.data_format == defs.data_format.tfrecord:

            # initialize tfrecord, check file consistency
            if self.do_training:
                self.input_source_files[defs.train_idx] = sett.input[defs.train_idx]
                if not os.path.exists(self.input_source_files[defs.train_idx]):
                    error("Input paths file does not exist: %s" % self.input_source_files[defs.train_idx])
                # read frames and classes
                self.read_frames_metadata()
                # set tfrecord
                self.input_source_files[defs.train_idx] =  self.input_source_files[defs.train_idx] + ".tfrecord"
                if not os.path.exists(self.input_source_files[defs.train_idx]):
                    error("Input file does not exist: %s" % self.input_source_files[defs.train_idx])

                self.reset_iterator(defs.phase.train)
                # count or get number of items
                self.get_input_data_count(defs.phase.train)
                # restore batch index to snapshot if needed - only makes sense in training
                if self.batch_index > 0:
                    self.fast_forward_iter(self.train_iterator)
                # calculate batches: just batchsizes per batch; all is in the tfrecord
                # calculate batches
                if self.batch_item == defs.batch_item.default:
                    num_whole_batches = self.num_items_train // self.batch_size_train
                    items_left = self.num_items_train - num_whole_batches * self.batch_size_train
                elif self.batch_item == defs.batch_item.clip:
                    num_whole_batches = sum(self.clips_per_video) // self.batch_size_train
                    items_left = sum(self.clips_per_video) - num_whole_batches * self.batch_size_train

                self.batches_train = [self.batch_size_train for _ in range(num_whole_batches)]
                if items_left:
                    self.batches_train.append(items_left)

            if self.do_validation:
                self.input_source_files[defs.val_idx] = sett.input[defs.val_idx] + ".tfrecord"
                if not os.path.exists(self.input_source_files[defs.val_idx]):
                    error("Input file does not exist: %s" % self.input_source_files[defs.val_idx])
                self.reset_iterator(defs.phase.val)
                # count or get number of items
                self.get_input_data_count(defs.phase.val)
                # calculate batches
                if self.batch_item == defs.batch_item.default:
                    num_whole_batches = self.num_items_val // self.batch_size_val
                    items_left = self.num_items_val - num_whole_batches * self.batch_size_val
                elif self.batch_item == defs.batch_item.clip:
                    num_whole_batches = sum(self.clips_per_video) // self.batch_size_val
                    items_left = sum(self.clips_per_video) - num_whole_batches * self.batch_size_val
                self.batches_val = [ self.batch_size_val for _ in range(num_whole_batches)]

                if items_left:
                    self.batches_val.append(items_left)
        elif self.data_format == defs.data_format.raw:
            # read input files
            self.input_source_files[defs.train_idx] = sett.input[defs.train_idx]
            self.input_source_files[defs.val_idx] = sett.input[defs.val_idx]

            # read frames and classes
            self.read_frames_metadata()
            if self.do_training:

                # for raw input mode, calculate batches for paths and labels
                imgs = sublist(self.frame_paths[defs.train_idx], self.batch_size_train)
                lbls = sublist(self.frame_classes[defs.train_idx], self.batch_size_train)
                for l in range(len(lbls)):
                    self.batches_train.append([
                        imgs[l],
                        list(map(int, lbls[l]))
                    ])
            if self.do_validation:
                imgs = sublist(self.frame_paths[defs.val_idx], self.batch_size_val)
                lbls = sublist(self.frame_classes[defs.val_idx], self.batch_size_val)
                for l in range(len(lbls)):
                    self.batches_val.append([
                        imgs[l],
                        list(map(int, lbls[l]))
                    ])
            self.num_items_train = len(self.frame_paths[defs.train_idx])
            self.num_items_val = len(self.frame_paths[defs.val_idx])

            self.num_items_train = len(self.frame_paths[defs.train_idx])
            self.num_items_val = len(self.frame_paths[defs.val_idx])
        else:
            error("Undefined data format [%s]" % self.data_format)

    # read or count how many data items are in training / validation
    def get_input_data_count(self,phase):

        idx = defs.train_idx if phase == defs.phase.train else defs.val_idx
        input_file = self.input_source_files[idx]
        # check if a .size file exists
        size_file = input_file + ".size"
        if not os.path.exists(size_file):
            # unsupported
            error("Could not file data size file: ",size_file)

        # read the data
        datainfo = read_file_dict(size_file)
        num_items = eval(datainfo['items'])
        if phase == defs.phase.train:
            self.num_items_train = num_items
        elif phase == defs.phase.val:
            self.num_items_val = num_items

        if datainfo['type'] != self.input_mode:
            error("Specified input mode %s but the size file contains %s" % (self.input_mode, datainfo['type']))

        cpv = eval(datainfo['cpi'])
        fpc = eval(datainfo['fpc'])

        assert ( (cpv is not None) != (self.input_mode == defs.input_mode.image) ), \
            "Read cpi: %s but input mode is %s" % (str(cpv), self.input_mode)

        assert ((fpc is not None) != (self.input_mode == defs.input_mode.image)), \
            "Read fpc of %d but input mode is %s" % (fpc, self.input_mode)


        # print data information
        if type(cpv)==list:
            # check consistency
            if len(cpv) != num_items:
                error("Read %d items but got cpv list of size %d" % (num_items, len(cpv)))

            # and len(self.clips_per_video) > 12:
            # if it's large, make it small and printable
            nump = 3
            cpv_str = "["+ str(cpv[:nump])[1:-1] +", ...," + str(cpv[-nump:])[1:-1] + "]"
        else:
            cpv_str = str(cpv)

        self.clips_per_video = cpv
        self.num_frames_per_clip = fpc
        self.max_caption_length = int(datainfo['labelcount'])
        self.max_sequence_length = self.max_caption_length + 1

        info("Read [%s] data, count: %d, cpv: %s, fpc: %s, type: %s, lblcount: %d+1" %
             (phase, num_items, cpv_str, str(self.num_frames_per_clip), self.input_mode, self.max_caption_length))

    # set iterator to point to the beginning of the tfrecord file, per phase
    def reset_iterator(self,phase):
        if not self.data_format == defs.data_format.tfrecord:
            return
        if phase == defs.phase.train:
            self.train_iterator = tf.python_io.tf_record_iterator(path=self.input_source_files[defs.train_idx])
        else:
            self.val_iterator = tf.python_io.tf_record_iterator(path=self.input_source_files[defs.val_idx])

    # move up an iterator
    def fast_forward_iter(self):

        # fast forward to batch index
        num_forward = self.batch_index
        if self.input_mode == defs.input_mode.image:
            info("Fast forwarding to the batch # %d ( image # %d)" % (self.batch_index+1, num_forward+1))
        elif self.input_mode == defs.input_mode.video:
            num_forward = num_forward * self.num_frames_per_clip
            info("Fast forwarding to the batch # %d ( image # %d )" % (self.batch_index + 1, num_forward + 1))


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
        info("Dataset batch information per epoch:" )
        header_fmt = "%-6s %-8s %-8s %-10s %-6s %-6s %-9s %-6s"
        values_fmt = "%-6s %-8s %-8s %-10d %-6d %-6d %-9s %-6d"
        info(header_fmt % ("Mode","bmode","items", "clips", "frames","b-size","b-num","b-index"))
        if self.do_training:
            items = self.num_items_train
            clips = 0 if self.clips_per_video is None else sum(self.clips_per_video)
            frames = items if self.num_frames_per_clip is None else clips * self.num_frames_per_clip
            info(values_fmt %
                 (defs.phase.train, self.batch_item, items, clips, frames,
                  self.batch_size_train, len(self.batches_train), self.batch_index_train))
        if self.do_validation:
            items = self.num_items_val
            clips = 0 if self.clips_per_video is None else sum(self.clips_per_video)
            frames = items if self.num_frames_per_clip is None else clips * self.num_frames_per_clip
            info(values_fmt %
                 (defs.phase.val, self.batch_item, items, clips, frames,
                  self.batch_size_val, len(self.batches_val),  self.batch_index_val))

    # get the batch size
    def get_batch_size(self):
        if self.do_training:
            return self.batch_size_train
        else:
            return self.batch_size_val

    # get the global step
    def get_global_batch_step(self):
        return self.epoch_index * len(self.batches) + self.batch_index

    # print iteration information
    def print_iter_info(self, num_images, num_labels):
        if self.phase == defs.phase.train:
            info("Mode: [%s], epoch: %2d/%2d, batch %4d / %4d : %3d images, %3d labels" %
                         (self.phase, self.epoch_index + 1,
                          self.epochs, self.batch_index, len(self.batches),
                          num_images, num_labels))
        # same as train, but no epoch
        elif self.phase == defs.phase.val:
            info("Mode: [%s], batch %4d / %4d : %3d images, %3d labels" %
                         (self.phase, self.batch_index, len(self.batches),
                          num_images, num_labels))

    # specify valid loop iteration
    def loop(self):
        return self.batch_index < len(self.batches)

    # check if testing should happen now
    def should_test_now(self):
        retval = self.do_validation and (self.batch_count > 0) and ((self.batch_count % self.validation_interval) == 0)
        self.batch_count = self.batch_count + 1
        return retval

    # check if there's only one clip per video
    def single_clip(self):
        if type(self.clips_per_video) == int:
            return self.clips_per_video == 1
        return False

    # check if we should save
    def should_save_now(self, global_step):
        if self.save_interval == None or self.phase != defs.phase.train:
            return False
        return global_step % self.save_interval == 0
