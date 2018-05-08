
# files and io
import os, pickle, math
import tqdm

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

    # single frame paths and classes, for raw input mode
    frames = []
    labels = []

    # run settings
    run_folder = None
    clips_per_video = None
    num_frames_per_clip = None
    frame_format = None
    data_format = None
    include_labels = False
    do_padding = False

    # input frames settings
    raw_image_shape = None
    desired_image_shape = None
    input_mode = None
    serialization_size = None

    # mean image subtraction
    mean_image = None

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
    num_items = 0

    batch_item = None

    # variables for validation clip batches
    video_index = 0

    # description
    embedding_matrix = None
    vocabulary = None
    max_caption_length = None
    max_sequence_length = None

    # mean subtraction
    def shoud_subtract_mean(self):
        return self.mean_image is not None

    # read paths to images
    def read_frames_metadata(self):
        # read paths and labels
        self.frames = []
        self.labels = []
        with open(self.path, 'r') as f:
            for path_label_str in f:
                path_labels = path_label_str.split()
                item_path = path_labels[0]
                item_labels = path_labels[1:]
                if self.prepend_folder is not None:
                    item_path = os.path.join(self.prepend_folder, item_path)
                self.frames.append(item_path)
                self.labels.append(item_labels)


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

    def deserialize_example(self, string_record):
        '''
        Deserialize single TFRecord example
        '''
        ser_size = len(string_record)
        if self.serialization_size:
            if ser_size != self.serialization_size:
                error("Encountered a different serialization size: %d vs stored %d" % \
                      (ser_size, self.serialization_size))
            else:
                self.serialization_size = ser_size
        example = tf.train.Example()
        example.ParseFromString(string_record)
        img_string = (example.features.feature['image_raw']
                      .bytes_list
                      .value[0])
        height = int(example.features.feature['height']
                     .int64_list
                     .value[0])
        width = int(example.features.feature['width']
                    .int64_list
                    .value[0])
        depth = (example.features.feature['depth']
                 .int64_list
                 .value[0])
        label = (example.features.feature['label']
                 .int64_list
                 .value)
        label = list(label)
        label = label[0] if len(label) == 0 else label

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        image = img_1d.reshape(height, width, depth)
        return image, label



    def deserialize_vector(self, num_vectors):
        vectors, labels = [], []
        for _ in range(num_vectors):
            try:
                string_record = next(self.iterator)
                example = tf.train.Example()
                example.ParseFromString(string_record)
                vec_string = (example.features.feature['vector_raw']
                    .bytes_list
                    .value[0])
                dim = int(example.features.feature['dimension']
                          .int64_list
                          .value[0])
                label = (example.features.feature['label']
                         .int64_list
                         .value)
                label = list(label)
                label = label[0] if len(label) == 0 else label
                vector = np.fromstring(vec_string, dtype=np.float32)
                if dim != len(vector):
                    error("Deserialized vector length %d but dimension stored is %d." % (len(vector), dim))

                vectors.append(vector)
                labels.append(label)

            except StopIteration:
                break
            except Exception as ex:
                warning(ex)
                error("Error reading tfrecord vector.")

        return vectors, labels

    # read from tfrecord
    def deserialize_from_tfrecord(self, images_per_iteration):
        # images_per_iteration :
        images, labels = [], []
        for imidx in range(images_per_iteration):
            try:
                string_record = ""
                string_record = next(self.iterator)
                image, label = self.deserialize_example(string_record)
            except StopIteration:
                if imidx < images_per_iteration:
                    image, label = self.reread_serialized(imidx)
                    #error('Encountered unexpected EOF while reading TFRecord example # %d in the batch.' % imidx)
                    #image, label = self.manually_read_image(imidx)

            except Exception as ex:
                warning('Encountered exception while reading TFRecord example # %d in the batch. Trying to print length:' % imidx)
                try:
                    print("String record length:",len(string_record))
                except:
                    print("Failed to print the length of the string record")

                warning('Retrying to parse the idx %d example a total of %d times' % (imidx, self.read_tries))
                counter = 0
                success = False
                while True:
                    try:
                        if counter > self.read_tries:
                            break
                        counter +=1
                        print("Try #%d" % counter)
                        image, label = self.deserialize_example(string_record)
                        success = True
                    except:
                        pass

                    if success:

                        info("Serialization error recovered through example reread!")
                    else:
                        image, label = self.reread_serialized(imidx)
                    break

            image = self.process_image(image)
            images.append(image)
            labels.append(label)

        return images, labels

    def reread_serialized(self, imidx):
        # try to reread the long way around: reset the iterator
        self.reset_iterator()
        # forward to the index within the current batch
        for _ in range(imidx):
            string_record = next(self.iterator)
        try:
            image, label = self.deserialize_example(string_record)
        except:
            error("Failed to troubleshoot serialization error.")
        info("Serialization error recovered: re-read through iterator restore!")
        return image, label

    # fallback when tfrecord image reads fail
    def manually_read_image(self, idx_in_batch):
        global_image_index = self.batch_index * self.batch_size + idx_in_batch
        impath = self.frames[global_image_index]
        info("Attempting to manually read global image index %d : %s" % (global_image_index, impath))
        image = self.read_image(impath)
        image = imresize(image, self.raw_image_shape)
        label = self.labels[global_image_index]
        label = list(map(int,label))
        return image,label

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
            if self.input_mode in [defs.input_mode.video, defs.input_mode.vectors]:
                images, labels = self.get_next_batch_video_tfr()
            elif self.input_mode == defs.input_mode.image:
                images, labels = self.get_next_batch_frame_tfr(currentBatch)

        if defs.workflows.is_description(self.workflow):

            if self.do_validation:
                # pad incomplete batch with zeros in both images and captions
                num_pad = self.batch_size_val - len(images)
                images.extend( [ np.zeros(images[0].shape, images[0].dtype) for _ in range(num_pad)] )
                labels.extend( [[-1] for _ in range(num_pad)])
            ground_truth = self.labels_to_words(labels)
        else:
            ground_truth = labels_to_one_hot(labels, self.num_classes)

        self.advance_batch_index()
        return images, ground_truth

    # get captions from logit vectors
    def validation_logits_to_captions(self, logits, start_index = 0):
        image_ids = []
        captions = []
        if self.eval_type == defs.eval_type.coco:
            # read the image ids
            parts = self.path.split(".")
            image_paths_file = ".".join(parts[0:-1])
            line_counter = -1
            chunk_size = len(logits)
            # read image ids, presumed to be the suffixless basename
            with open(image_paths_file, "r") as fp:
                for line in fp:
                    line_counter += 1
                    # make sure you read the correct chunk
                    if line_counter < start_index:
                        continue
                    # limit to chunk length
                    if len(image_ids) == len(logits):
                        break
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
            debug("Read image ids %d-%d from %s" % (start_index, start_index + chunk_size, os.path.basename(image_paths_file)))

        # generate captions from logits
        # recall that logits contain caption word indexes in the vocabulary
        for i,image_logits in enumerate(logits):
            image_caption = []
            for caption_index in image_logits:
                image_caption.append(self.vocabulary[caption_index])
            if not image_caption:
                image_caption = ' '
            captions.append(" ".join(image_caption))

        # Populate the return data
        return_data = [{"image_id":iid , "caption": caption} for (iid, caption) in zip(image_ids,captions)]
        debug("Generated ids:")
        for i in range(len(image_ids)):
            debug("image id: %s caption:%s" % (str(image_ids[i]),str(captions[i])))
        return return_data

    def apply_caption_padding(self, word_vectors, curr_caption_length, batch_index):
        # pad to the max number of words with zero vectors. Also keep track of non-pad entries
        num_left_to_max = self.max_sequence_length - curr_caption_length
        local_indexes, offset = list(range(curr_caption_length)), batch_index * (self.max_sequence_length)
        no_pad_index = [offset + elem for elem in  local_indexes]
        if num_left_to_max > 0:
            word_vectors= np.vstack((word_vectors, np.zeros([num_left_to_max, word_vectors.shape[-1]], np.float32)))
        return no_pad_index, word_vectors

    # get word embedding vectors from vocabulary encoded indices
    def labels_to_words(self, raw_batch_labels):
        # initialize to empties
        batch_word_vectors = np.zeros([0, self.embedding_matrix.shape[1]], np.float32)
        batch_labels = np.zeros([0, self.num_classes], np.int32)
        batch_no_pad_index = []

        # get special tokens indices
        bos_index = self.vocabulary.index("BOS")
        eos_index = self.vocabulary.index("EOS")
        # for each label set
        for batch_index, item_labels in enumerate(raw_batch_labels):
            # debug print the labels
            input_labels = [bos_index] + item_labels
            output_labels = item_labels + [eos_index]
            debug("Item io labels:  %s, %s" % (str(input_labels), str(output_labels)))

            # get caption word embedding vectors - set BOS as first input token
            word_vectors = self.embedding_matrix[bos_index, :]
            # if training, put the caption word vectors after the BOS in the input, else leave just BOS
            if self.include_labels:
                word_vectors = np.vstack((word_vectors, self.embedding_matrix[item_labels, :]))

            # get labels needed for training loss / validation accuracy
            # there is no need to onehot-encode in the validation case, but it's done to get uniform-length labels
            labels = np.asarray(labels_to_one_hot(item_labels, self.num_classes))
            if self.include_labels:
                # in training mode, append with EOS
                eos_onehot = labels_to_one_hot(eos_index, self.num_classes)
                labels = np.vstack((labels, eos_onehot))

            # pad the input caption to the maximum caption length.
            if self.do_padding:
                no_pad_index, word_vectors = self.apply_caption_padding(word_vectors, len(input_labels), batch_index)
                batch_no_pad_index.extend(no_pad_index)

            # append labels and word vectors to their batch containers
            batch_labels = np.vstack((batch_labels, labels))
            batch_word_vectors = np.vstack((batch_word_vectors, word_vectors))

        # word vectors batch always contains max_caption_len x batch-size elements, due to padding.
        # batch labels contain the exact labels for each word for each image caption. I.e., no padding there.
        ground_truth = {}
        ground_truth['word_embeddings'] = batch_word_vectors
        ground_truth['onehot_labels'] = batch_labels
        ground_truth['caption_lengths'] = list(map(len, raw_batch_labels))
        ground_truth['non_padding_index'] = batch_no_pad_index
        debug("Non-padding index in batch: %s" % str(batch_no_pad_index))
        return ground_truth

    def get_next_batch_video_tfr(self):
        if self.batch_item == defs.batch_item.default:
            # batch size refers to videos. Get num clips per videos in batch
            curr_video = self.batch_index * self.batch_size
            curr_cpv = self.clips_per_video[curr_video : curr_video + self.batch_size]
            num_frames_in_batch = sum([self.num_frames_per_clip * x for x in curr_cpv])
            if not num_frames_in_batch:
                error("Computed 0 frames in next batch.")
            # read the frames from the TFrecord serialization file
            if self.input_mode ==  defs.input_mode.vectors:
                images, labels_per_frame = self.deserialize_vector(num_frames_in_batch)
            else:
                images, labels_per_frame = self.deserialize_from_tfrecord(num_frames_in_batch)

            # limit to <numclips> labels per video.
            fpv = [self.num_frames_per_clip * clip for clip in curr_cpv]
            fpv = list(np.cumsum(fpv))
            first_videoframe_idx = [0] + fpv[:-1]
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
            images, labels_per_frame = self.deserialize_from_tfrecord(num_frames_in_batch)
            # limit to 1 label per clip
            labels = labels_per_frame[0:: self.num_frames_per_clip]

        return images,labels

    def get_next_batch_frame_tfr(self, num_frames_in_batch):
        # read images from a TFrecord serialization file
        images, labels = self.deserialize_from_tfrecord(num_frames_in_batch)
        if self.input_mode == defs.input_mode.video:
            labels = [labels[l] for l in range(0, len(labels), self.num_frames_per_clip)]
        return images,labels

    def advance_batch_index(self):
        self.batch_index += 1

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
    def crop_image(self, image, mode):
        if not self.raw_image_shape:
            # compute them
            h, w = self.compute_crop(image.shape, self.desired_image_shape, mode)
        else:
            h, w = self.crop_h, self.crop_w
        if mode == defs.imgproc.rand_crop:
            # select crop idx
            h = choice(h)
            w = choice(w)
        h, w = int(h), int(w)

        # apply crop
        image = image[
                h:h + self.desired_image_shape[0],
                w:w + self.desired_image_shape[1],
                :]
        return image


    # read image from disk
    def read_image(self, imagepath):
        image = imread(imagepath)

        debug("Reading image %s" % imagepath)
        # for grayscale images, duplicate
        # intensity to color channels
        if len(image.shape) <= 2:
            image = np.repeat(image[:, :, np.newaxis], 3, 2)
        # drop channels other than RGB
        image = image[:, :, :3]
        #  convert to BGR
        image = image[:, :, ::-1]
        image = self.process_image(image)
        return image

    # apply image post-process
    def process_image(self, image):
        # resize to desired raw dimensions, if defined
        if defs.imgproc.raw_resize in self.imgproc:
            image = imresize(image, self.raw_image_shape)
        # crop or resize to network input size
        if defs.imgproc.rand_crop in self.imgproc:
            image = self.crop_image(image, defs.imgproc.rand_crop)
        elif defs.imgproc.center_crop in self.imgproc:
            image = self.crop_image(image, defs.imgproc.center_crop)
        elif defs.imgproc.resize in self.imgproc:
            image = imresize(image, self.desired_image_shape)

        if not image.shape == self.desired_image_shape:
            error("Encountered image shape %s but desired shape is %s" % (image.shape, self.desired_image_shape))
        if defs.imgproc.sub_mean in self.imgproc:
            image = image - self.mean_image

        if defs.imgproc.rand_mirror in self.imgproc:
            if not randrange(2):
                image = image[:,::-1,:]
        return image

    def initialize(self, id, path, mean_image, prepend_folder, desired_image_shape, imgproc, raw_image_shape,
                   data_format, frame_format, batch_item, num_classes, tag, read_tries):
        info("Initializing dataset [%s]" % id)
        self.id = id
        self.path = path
        self.data_format = data_format
        self.frame_format = frame_format
        self.prepend_folder = prepend_folder
        self.mean_image = mean_image
        self.desired_image_shape = desired_image_shape
        self.imgproc = imgproc
        self.batch_item = batch_item
        self.raw_image_shape = raw_image_shape
        self.num_classes = num_classes
        self.tag = tag
        self.read_tries = read_tries


    def build_mean_image(self):
        if defs.imgproc.sub_mean in self.imgproc:
            # build the mean image
            height = self.desired_image_shape[0]
            width = self.desired_image_shape[1]
            blue = np.full((height, width), self.mean_image[0])
            green = np.full((height, width), self.mean_image[1])
            red = np.full((height, width), self.mean_image[2])
            self.mean_image = np.ndarray.astype(np.stack([blue, green, red]),np.float32)
            self.mean_image = np.transpose(self.mean_image, [1, 2, 0])

    # restore from a checkpoint
    #def restore(self, batch_index, epoch_index, sequence_length):
    def restore(self, batch_index, epoch_index):
        self.batch_index = batch_index
        self.epoch_index = epoch_index
        #self.sequence_length = sequence_length
        self.fast_forward_iter()

    def initialize_imgproc(self):
        if self.input_mode == defs.input_mode.vectors:
            if self.imgproc:
                info("Ignoring imgproc due to input mode: [%s]" % self.input_mode)
            self.imgproc = []
            return

        self.build_mean_image()

        if defs.imgproc.rand_crop in self.imgproc:
            if self.raw_image_shape is None:
                warning("Random cropping without a fixed raw image shape.")
            else:
                # precompute available crops
                self.crop_h, self.crop_w = self.compute_crop(self.raw_image_shape, self.desired_image_shape, defs.imgproc.rand_crop)
        elif defs.imgproc.center_crop in self.imgproc:
            if self.raw_image_shape is None:
                warning("Center cropping without a fixed raw image shape.")
            else:
                # precompute the center crop
                self.crop_h, self.crop_w = self.compute_crop(self.raw_image_shape, self.desired_image_shape, defs.imgproc.center_crop)

    def compute_dataset_portion(self, freq_per_epoch, epochs):
        # set save interval
        save_interval = math.ceil(len(self.batches) / freq_per_epoch)
        num_saves = math.ceil(freq_per_epoch * epochs)
        info("Computed batch save interval (from %2.4f per %d-batched epoch) to %d batches and %d total saves" %
             (freq_per_epoch, len(self.batches), save_interval, num_saves))
        return save_interval, num_saves

    # compute horizontal and vertical crop range
    def compute_crop(self, raw_image_shape, image_shape, mode):
        if mode == defs.imgproc.center_crop:
            return tuple([np.floor((x[0]-x[1])/2) for x in zip(raw_image_shape, image_shape)])[:2]
        elif mode == defs.imgproc.rand_crop:
            crop_h = [i for i in range(0, raw_image_shape[0] - image_shape[0] - 1)]
            crop_w = [i for i in range(0, raw_image_shape[1] - image_shape[1] - 1)]
        return crop_h, crop_w


    # initialize run mode specific data
    def initialize_workflow(self, word_embeddings_file):
        if defs.workflows.is_description(self.workflow):
            # read embedding matrix
            info("Reading embedding matrix from file [%s]" % word_embeddings_file)
            self.vocabulary = []
            vectors = []
            with open(word_embeddings_file,"r") as f:
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
            # if a max caption length is read, assign it
            if self.sequence_length is not None:
                info("Restricting to a caption length of", self.max_caption_length)
                self.max_caption_length = self.sequence_length

    def get_embedding_dim(self):
        return int(self.embedding_matrix.shape[-1])

    def calculate_batches(self, batch_size, input_mode):
        self.batch_size = batch_size
        self.input_mode = input_mode
        # do initialization
        # read the data
        if not os.path.exists(self.path):
            error("Dataset path does not exist: %s" % self.path)
        self.read_frames_metadata()
        if self.data_format == defs.data_format.tfrecord:
            # update data path
            self.path += ".tfrecord"
            if not os.path.exists(self.path):
                error("TFRecord file path does not exist: %s" % self.path)
            # reset iterator
            self.reset_iterator()
        # read frames, classes, metadata
        self.get_input_data_count()
        self.initialize_imgproc()
        # calculate batches
        if self.batch_item == defs.batch_item.default:
            num_whole_batches = self.num_items // self.batch_size
            items_left = self.num_items - num_whole_batches * self.batch_size
        elif self.batch_item == defs.batch_item.clip:
            num_whole_batches = sum(self.clips_per_video) // self.batch_size
            items_left = sum(self.clips_per_video) - num_whole_batches * self.batch_size

        self.batches = [self.batch_size for _ in range(num_whole_batches)]
        if items_left:
            self.batches.append(items_left)

        self.tell()

    # run data-related initialization pre-run
    def initialize_data(self, sett):
        info("Reading [%s] data on input mode [%s]." % (self.data_format, self.input_mode))
        if self.data_format == defs.data_format.tfrecord:

            # initialize tfrecord, check file consistency
            if self.do_training:
                self.input_source_files[defs.train_idx] = sett.input_files[defs.train_idx]
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
                self.input_source_files[defs.val_idx] = sett.input_files[defs.val_idx] + ".tfrecord"
                if not os.path.exists(self.input_source_files[defs.val_idx]):
                    error("Input file does not exist: %s" % self.input_source_files[defs.val_idx])
                self.reset_iterator(defs.phase.val)
                # count or get number of items
                self.get_input_data_count(defs.phase.val)
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
    def get_input_data_count(self):
        # check if a .size file exists
        size_file = self.path + ".size"
        if not os.path.exists(size_file):
            # unsupported
            error("Could not file data size file: ",size_file)

        # read the data
        datainfo = read_file_dict(size_file)
        self.num_items = eval(datainfo['items'])

        if datainfo['type'] != defs.input_mode.vectors:
            if self.input_mode is not None and (datainfo['type'] != self.input_mode):
                error("Specified input mode is [%s] but the size file contains [%s]" % (self.input_mode, datainfo['type']))
        else:
            self.input_mode = defs.input_mode.vectors

        cpv = eval(datainfo['cpi'])
        fpc = eval(datainfo['fpc'])

        assert ( (cpv is not None) != (self.input_mode == defs.input_mode.image) ), \
            "Read cpi: %s but input mode is %s" % (str(cpv), self.input_mode)

        assert ((fpc is not None) != (self.input_mode == defs.input_mode.image)), \
            "Read fpc of %d but input mode is %s" % (fpc, self.input_mode)

        # print data information
        if type(cpv)==list:
            # read RLC-encoded cpv, if applicable
            if type(cpv[0]) == tuple:
                cpv= [item for num, item in cpv for _ in range(num)]
            if len(cpv) != self.num_items:
                error("Read %d items but got cpv list of size %d" % (self.num_items, len(cpv)))

            # and len(self.clips_per_video) > 12:
            # if it's large, make it small and printable
            nump = 3
            cpv_str = "["+ str(cpv[:nump])[1:-1] +", ..., " + str(cpv[-nump:])[1:-1] + "]"
        else:
            cpv_str = str(cpv)

        self.clips_per_video = cpv
        self.num_frames_per_clip = fpc
        loaded_caption_length = int(datainfo['labelcount'])
        # if a max caption length was already read, make sure it is greq the one in the data
        if self.max_caption_length is not None:
            if loaded_caption_length > self.max_caption_length:
                error("Data contains a max caption length of %d, but current setting is restricted to %d" % 
                      (loaded_caption_length, self.max_caption_length))
        else:
            self.max_caption_length = loaded_caption_length
                # if the loaded caption length is leq the restricted, keep the latter
        self.max_sequence_length = self.max_caption_length + 1

        info("Read [%s] data, count: %d, cpv: %s, fpc: %s, type: %s, lblcount: %d" %
             (str(self.id), self.num_items, cpv_str, str(self.num_frames_per_clip), self.input_mode, self.max_caption_length))

    # set iterator to point to the beginning of the tfrecord file, per phase
    def reset_iterator(self):
        if not self.data_format == defs.data_format.tfrecord:
            return
        if self.iterator:
            self.iterator.close()
        self.iterator = tf.python_io.tf_record_iterator(path=self.path)

    # reset to the start of a dataset
    def rewind(self):
        self.reset_iterator()
        self.batch_index = 0

    # move up an iterator
    def fast_forward_iter(self, index = None):
        # fast forward to batch index
        if index is None:
            num_forward = self.batch_index
        else:
            num_forward = index

        num_all_frames = sum([ x * y * self.num_frames_per_clip for (x, y) in zip(self.batches, self.clips_per_video)])
        if self.input_mode == defs.input_mode.image:
            error("Image fast-forwarding not implemented yet")
        elif self.input_mode == defs.input_mode.video:
            # if the max number of batches is specified, account for potentially
            # incomplete last batch
            if num_forward == len(self.batches):
                num_forward = num_all_frames
            else:
                # get how many items we have to move up to
                item_index = self.batch_index * self.batch_size
                if self.batch_item == defs.batch_item.default:
                    # if the batch size refers to whole items, gotta count the clips
                    # up to the restoration point (for variable cpv cases)
                    num_clips = 0
                    for i in range(len(self.clips_per_video)):
                        if i == item_index:
                            break
                        num_clips += self.clips_per_video[i]
                    num_forward = num_clips * self.num_frames_per_clip
                elif self.batch_item == defs.batch_item.clip:
                    # if it refers to clips, it's straightforward
                    num_forward = self.num_frames_per_clip * item_index


        info("Fast forwarding to batch # %d/%d ( image # %d/%d )" % (self.batch_index + 1, len(self.batches), num_forward + 1, num_all_frames))
        with tqdm.tqdm(total=num_forward, ascii=True) as pbar:
            for _ in range(num_forward):
                next(self.iterator)
                pbar.update()

    # print active settings
    def tell(self):
        info("Dataset batch information per epoch:" )
        header_fmt = "%-8s %-8s %-10s %-6s %-6s %-8s %-8s %-7s"
        values_fmt = "%-8s %-8s %-10d %-6d %-6d %-8s %-8d %-7s"
        info(header_fmt % ("bmode","items", "clips", "frames","b-size","b-num","b-index","imgprc"))
        imgproc = defs.imgproc.to_str(self.imgproc)
        items = self.num_items
        clips = 0 if self.clips_per_video is None else sum(self.clips_per_video)
        frames = items if self.num_frames_per_clip is None else clips * self.num_frames_per_clip
        info(values_fmt %
             ( self.batch_item, items, clips, frames,
              self.batch_size, len(self.batches), self.batch_index, imgproc))

    # get the global step
    def get_global_batch_step(self):
        return self.epoch_index * len(self.batches) + self.batch_index

    # specify valid loop iteration
    def loop(self):
        return self.batch_index < len(self.batches)

    # check if there's only one clip per video
    def single_clip(self):
        if type(self.clips_per_video) == int:
            return self.clips_per_video == 1
        return False
    def get_image_shape(self):
        if self.desired_image_shape is None:
            return self.raw_image_shape
        return self.desired_image_shape

