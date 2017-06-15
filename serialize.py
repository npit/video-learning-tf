import tensorflow as tf
import numpy as np
from random import shuffle
from scipy.misc import imread, imresize, imsave

import logging, time, threading, os
from utils_ import *
import matplotlib.pyplot as plt
import configparser
import sys

'''
Script for production of training / testing data collections and serialization to tf.record files.
'''

init_file = "config.ini"

# necessary config. variables
input_files = []

# defaults
path_prepend_folder = None
num_threads = 4
num_items_per_thread = 500
num_frames_per_clip = 16
raw_image_shape = (240,320,3)
clipframe_mode = defs.clipframe_mode.rand_clips
clip_offset_or_num = 1
frame_format = "jpg"
force_video_metadata = False
do_shuffle = False
do_serialize = False

# initialize from file
def initialize_from_file(init_file):
    if init_file is None:
        return
    if not os.path.exists(init_file):
        return
    tag_to_read = "serialize"
    print("Initializing from file %s" % init_file)
    config = configparser.ConfigParser()
    config.read(init_file)
    if not config[tag_to_read ]:
        error('Expected header [%s] in the configuration file!' % tag_to_read)

    config = config[tag_to_read]
    return config


# datetime for timestamps
def get_datetime_str():
    #return time.strftime("[%d|%m|%y]_[%H:%M:%S]")
    return time.strftime("%d%m%y_%H%M%S")

# configure logging settings
def configure_logging():
    logging_level = logging.INFO

    logfile = "log_serialize_" + get_datetime_str() + ".log"
    print("Using logfile: %s" % logfile)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging_level)

    formatter = logging.Formatter('%(asctime)s| %(levelname)7s - %(filename)15s - line %(lineno)4d - %(message)s')

    # # file handler
    handler = logging.FileHandler(logfile)
    handler.setLevel(logging_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(consoleHandler)
    return logger


logger = configure_logging()
# helper tfrecord function
def _int64_feature( value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# helper tfrecord function
def _bytes_feature( value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_multithread(clips_per_vid_or_item_paths ,frame_paths, labels, outfile, mode):

    # first of all, write the number of items and the image size in the tfrecord
    with open(outfile + ".size","w") as f:
        f.write("%d\n" % len(clips_per_vid_or_item_paths))
        if mode == defs.input_mode.video or force_video_metadata:
            f.write("%d\n" % num_frames_per_clip)
            if mode == defs.input_mode.video:
                if clipframe_mode == defs.clipframe_mode.iterative:
                    # write num clips per video, as it may vary
                    for numclips in clips_per_vid_or_item_paths:
                        f.write("%d " % numclips)
                else:
                    # constant number of clips
                    f.write("%d\n" % clip_offset_or_num)
        else:
            # image mode
            frame_paths = clips_per_vid_or_item_paths

    # split up paths/labels list per thread run
    num_images_per_thread_run = num_items_per_thread * num_threads
    paths_per_thread_run = sublist(frame_paths, num_images_per_thread_run)
    labels_per_thread_run = sublist(labels, num_images_per_thread_run)


    count = 0
    writer = tf.python_io.TFRecordWriter(outfile)
    for run_index in range(len(paths_per_thread_run)):

        paths_in_run = paths_per_thread_run[run_index]
        labels_in_run = labels_per_thread_run[run_index]

        tic = time.time()
        logger.debug("Processing %d items for the run." % len(paths_in_run))

        paths_per_thread = sublist(paths_in_run, num_items_per_thread )
        labels_per_thread = sublist(labels_in_run, num_items_per_thread )

        logger.debug("Items scheduled list len : %d." % (len(paths_per_thread)))

        num_threads_in_run = len(paths_per_thread)
        for t in range(num_threads_in_run):
            logger.debug("Frames scheduled for thread #%d : %d." % (t, len(paths_per_thread[t])))
        # start threads
        threads = [[] for _ in range(num_threads_in_run)]
        frames =  [[] for _ in range(num_threads_in_run)]
        for t in range(num_threads_in_run):
            threads[t] = threading.Thread(target=read_item_list_threaded,args=(paths_per_thread[t],frames,t))
            threads[t].start()


        # wait for threads to read
        for t in range(num_threads_in_run):
            threads[t].join()

        for t in range(num_threads_in_run):
            logger.debug("Frames produced  for thread #%d : %d." % (t, len(frames[t])))


        # write the read images to the tfrecord



        for t in range(num_threads_in_run):
            if not frames[t]:
                logger.error("Thread # %d encountered an error." % t)
                exit(1)
            serialize_to_tfrecord(frames[t], labels_per_thread[t], outfile, writer)
            count += len(frames[t])


        logger.info("Processed %d frames, latest %d-sized batch took %s." %
                    (count, sum(list(map(len,paths_per_thread))), elapsed_str(time.time()-tic)))

    writer.close()


def get_item_paths(paths_list, mode):
    logger.info("Generating paths...")
    tic = time.time()
    paths_per_video = []
    if mode == defs.input_mode.image:
        return paths_list
    else:
        for vid_idx in range(len(paths_list)):
            #logger.info("Processing path %d / %d" % (vid_idx+1, len(paths_list)))
            video_path = paths_list[vid_idx]
            video_frame_paths = get_video_frame_paths(video_path)
            paths_per_video.append(video_frame_paths)
    logger.info("Total path generation time: %s " % elapsed_str(time.time() - tic))            
    return paths_per_video



def read_item_list_threaded(paths, storage, id):
    for framepath in paths:
        image = read_image(framepath)
        if image is None:
            return
        storage[id].append(image)


def serialize_to_tfrecord( frames, labels, outfil, writer):
    for idx in range(len(frames)):
        frame = frames[idx]
        label = labels[idx]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(raw_image_shape[0]),
            'width': _int64_feature(raw_image_shape[1]),
            'depth': _int64_feature(raw_image_shape[2]),
            'label': _int64_feature(int(label)),
            'image_raw': _bytes_feature(frame.tostring())}))
        writer.write(example.SerializeToString())

# read all frames for a video
def get_video_frame_paths(path):
    frames_indexes = []

    files = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    files = sorted(files)
    num_files = len(files)

    # generate a number of frame paths from the video path
    if clipframe_mode == defs.clipframe_mode.rand_frames:

        # select frames randomly from the video
        avail_frames = list(range(num_files))
        shuffle(avail_frames)
        avail_frames = avail_frames[:num_frames_per_clip]
        frames_indexes.extend(avail_frames)

    elif clipframe_mode == defs.clipframe_mode.rand_clips:

        # get <num_clips> random chunks of a consequtive <num_frames> frames
        possible_chunk_start = list(range(num_files - num_frames_per_clip + 1))
        if len(possible_chunk_start) < clip_offset_or_num:
            error("Video %s cannot sustain a number of %d unique %d-frame clips" %
                  (path, clip_offset_or_num, num_frames_per_clip))
        shuffle(possible_chunk_start)
        for _ in range(clip_offset_or_num):
            start_index = possible_chunk_start[-1]
            possible_chunk_start = possible_chunk_start[:-1]
            clip_frames = list(range(start_index, start_index + num_frames_per_clip))
            frames_indexes.extend(clip_frames)

    elif  clipframe_mode == defs.clipframe_mode.iterative:
        # get all possible video clips.
        start_indexes = list(range(0 , num_files - num_frames_per_clip + 1, num_frames_per_clip + clip_offset_or_num))
        for s in start_indexes :
            frames_indexes.extend(list(range(s,s+num_frames_per_clip )))

    frame_paths = []
    for fridx in frames_indexes:
        frame_path = os.path.join(path, files[fridx])
        frame_paths.append(frame_path)
    return frame_paths

 # read image from disk

def read_image(imagepath):
    try:
        image = imread(imagepath)
        logger.debug("Reading image %s" % imagepath)
        # for grayscale images, duplicate
        # intensity to color channels
        if len(image.shape) <= 2:
            image = np.repeat(image[:, :, np.newaxis], 3, 2)
        # drop channels other than RGB
        image = image[:,:,:3]
        #  convert to BGR
        image = image[:, :, ::-1]
        # resize
        image = imresize(image, raw_image_shape)

        # there is a problem if we want to store mean-subtracted images, as we'll have to store a float per pixel
        # => 4 x the space of a uint8 image
        # image = image - mean_image
    except Exception as ex:
        logger.error("Error :" + str(ex))
        error("Error reading image.")
        return None
    return image


def display_image(image,label=None):
    print(label)
    plt.title(label)
    plt.imshow(image)
    plt.show()
    # plt.waitforbuttonpress()


# read from tfrecord
def deserialize_from_tfrecord( iterator, images_per_iteration):
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
            image = img_1d.reshape((raw_image_shape[0], raw_image_shape[1], raw_image_shape[2]))

            images.append(image)
            labels.append(label)

        except StopIteration:
            break
        except Exception as ex:
            logger.error('Exception at reading image, loading from scratch')
            logger.error(ex)
            error("Error reading tfrecord image.")

    return images, labels



def read_file(inp):
    mode = None
    logger.info("Reading input file %s " % (inp))

    paths = []
    labels = []
    with open(inp, 'r') as f:
        for line in f:

            path, label = line.split(' ')
            path = path.strip()

            if mode is None:
                if path.lower().endswith("." + frame_format.lower()):
                    mode = defs.input_mode.image
                    logger.info("Set input mode to frames from paths-file items suffixes.")
                else:
                    mode = defs.input_mode.video
                    strlen = min(len(path), len(frame_format) + 1)
                    suffix = path[-strlen:]
                    logger.info(
                        "Set input mode to videos since paths-file item suffix [%s] differs from image format [%s]." % (
                        suffix, frame_format))

            if path_prepend_folder is not None:
                path = os.path.join(path_prepend_folder, path)
            paths.append(path)
            labels.append(int(label.strip()))
    return paths, labels, mode


def shuffle_pair(*args):
    z = list(zip(*args))
    shuffle(z)
    args= zip(*z)
    return args

def shuffle_paths(item_paths, paths, labels, mode):
    logger.info("Shuffling data...")

    if mode == defs.input_mode.image:
        item_paths, labels = shuffle_pair(item_paths, labels)
        return item_paths, labels

    # outer shuffle, of video order
    item_paths, paths, labels = shuffle_pair(item_paths, paths,labels)


    # inner shuffle, of frames
    if  clipframe_mode == defs.clipframe_mode.rand_frames:

        for vid_idx in range(len(paths)):
            p,l = zip(paths[vid_idx], labels[vid_idx])
            paths[vid_idx] = p
            labels[vid_idx] = l
        logger.info("Done shuffling.")
        return paths, labels
    else:
        # here we can only shuffle the clips themselves, not the frames within
        new_paths = []
        for vid_idx in range(len(paths)):
            # partition the paths to clips
            p = sublist(paths[vid_idx], num_frames_per_clip)
            # randomize the order of the clips
            shuffle(p)
            new_paths.append(p)
        paths = tuple(new_paths)
        logger.info("Done shuffling.")
        return item_paths, paths, labels

def write():
    framepaths_per_input = []
    for idx in range(len(input_files)):
        inp = input_files[idx]
        item_paths, item_labels, mode = read_file(inp)

        if mode == defs.input_mode.image:
            if do_shuffle:
                item_paths, item_labels = shuffle_paths(item_paths, None, item_labels, mode)
            paths_to_serialize, labels_to_serialize = item_paths, item_labels
            clips_per_video_or_item_paths = item_paths
            framepaths_per_input.append([item_paths, item_labels, None, None, mode])


        if mode == defs.input_mode.video:
            # generate paths per video
            paths = get_item_paths(item_paths, mode)
            if do_shuffle:
                item_paths, paths, item_labels = shuffle_paths(item_paths, paths, item_labels, mode)
            clips_per_video_or_item_paths = [ len(vid) for vid in paths ]

            # flatten: frame, for video in videos for frame in video
            labels_to_serialize = []
            for idx in range(len(item_labels)):
                ll = [item_labels[idx] for clip in paths[idx] for _ in clip]
                labels_to_serialize.extend(ll)
            paths_to_serialize = [ p for video in paths for clip in video for p in clip]
            framepaths_per_input.append([item_paths, item_labels, paths_to_serialize, labels_to_serialize, mode])

        if do_serialize:
            tic = time.time()
            output_file = inp + ".tfrecord"
            logger.info("Serializing %s " % (output_file))
            serialize_multithread(clips_per_video_or_item_paths, paths_to_serialize, labels_to_serialize, output_file , mode)
            logger.info("Done serializing %s " % inp)
            logger.info("Total serialization time: %s " % elapsed_str(time.time() - tic))
        logger.info("Done processing input file %s" % inp)
    return framepaths_per_input

# verify the serialization validity
def validate(written_data):

    for index in range(len(input_files)):
        inp = input_files[index]

        item_paths, item_labels, paths, labels, mode,  = written_data[index]
        if mode == defs.input_mode.image:
            paths = item_paths
            labels = item_labels
        num_validate = 10000 if len(paths) >= 10000 else len(paths)
        error_free = True
        idx_list = [ i for i in range(len(paths))]
        shuffle(idx_list)
        idx_list = idx_list[:num_validate]
        idx_list.sort()
        lidx = 0
        testidx = idx_list[lidx]
        iter = tf.python_io.tf_record_iterator(inp + ".tfrecord")
        for i in range(len(paths)):
            if not i == testidx:
                next(iter)
                continue

            frame = read_image(paths[i])
            label = labels[i]

            fframetf, llabeltf = deserialize_from_tfrecord(iter,1)
            frametf = fframetf[0]
            labeltf = llabeltf[0]

            if not np.array_equal(frame , frametf):
                logger.error("Unequal image @ %s" % paths[i])
                error_free = False
            if not label == labeltf:
                logger.error("Unequal label @ %s. Found %d, expected %d" % ( paths[i], label, labeltf))
                error_free = False

            lidx = lidx + 1
            if lidx >= len(idx_list):
                break

            testidx = idx_list[lidx]
        if not error_free:
            logger.error("errors exist.")
        else:
            logger.info("Validation for %s ok" % (inp + ".tfrecord"))


def write_paths_file(data):
    # write the selected clips / frames
    for i in range(len(data)):
        inp = input_files[i]

        item_paths, item_labels, paths, labels, mode = data[i]

        if do_shuffle:
            # write videos, if they got shuffled
            item_outfile = inp + ".shuffled"
            logger.info("Documenting shuffled video order to %s" % (item_outfile))
            with open(item_outfile,'w') as f:
                for v in range(len(item_paths)):
                    item = item_paths[v]
                    f.write("%s %d\n" % (item, item_labels[v]))


        clip_info = "" if clipframe_mode == defs.clipframe_mode.rand_frames or mode == defs.input_mode.image else ".%d.cpv" % clip_offset_or_num
        outfile = "%s%s.%dfpc.%s" % (inp, clip_info, num_frames_per_clip, defs.clipframe_mode.str(clipframe_mode))

        if not mode == defs.input_mode.video:
            continue
        logger.info("Documenting selected paths from file %s \n\tto %s" % (inp, outfile))
        with open(outfile, "w") as f:
            for path, label in zip(paths, labels):
                f.write("%s %d\n" % (path, label))

if len(sys.argv) > 1:
    init_file = sys.argv[-1]

keyvals = initialize_from_file(init_file)

for key in keyvals:
    exec("%s=%s" % (key, keyvals[key]))
print("Successfully initialized from file %s" % init_file)

# outpaths is either the input frame paths in image mode, or the expanded frame paths in video mode
written_data = write()
write_paths_file(written_data)
if do_serialize:
    validate(written_data)


