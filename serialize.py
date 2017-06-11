import tensorflow as tf
import numpy as np
from random import shuffle
from scipy.misc import imread, imresize, imsave

import logging, time, threading, os
from utils_ import *
import matplotlib.pyplot as plt
import configparser
import sys


init_file = "config.ini"

# necessary config. variables
input_files = []

# defaults
path_prepend_folder = None
num_threads = 4
num_items_per_thread = 500
raw_image_shape = (240,320,3)
video_frame_mode = defs.video_frame_mode.random
num_frames_per_clip = 16
num_clips_per_video = 1
frame_format = "jpg"
force_video_metadata = False

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


def serialize_multithread(paths, labels, outfile, mode):

    # first of all, write the number of items and the image size in the tfrecord
    with open(outfile + ".size","w") as f:

        f.write("%d\n" % len(paths))
        if mode == defs.input_mode.video or force_video_metadata:
            f.write("%d\n" % num_frames_per_clip)
            f.write("%d\n" % num_clips_per_video)

    # split up paths/labels list per thread run
    num_images_per_thread_run = num_items_per_thread * num_threads
    paths_per_thread_run = sublist(paths, num_images_per_thread_run)
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
        output_paths = [[] for _ in range(num_threads_in_run)]
        for t in range(num_threads_in_run):
            threads[t] = threading.Thread(target=read_item_list_threaded,args=(paths_per_thread[t],mode,frames,t, output_paths))
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
            if mode == defs.input_mode.video:
                duplist = [[label for _ in range(num_frames_per_clip * num_clips_per_video)] for label in labels_per_thread[t]]
                labels_per_thread[t] = []
                for l in duplist:
                    labels_per_thread[t].extend(l)

            serialize_to_tfrecord(frames[t], labels_per_thread[t], outfile, writer)
            count += len(frames[t])


        logger.info("Processed %d frames, latest %d-sized batch took %s." %
                    (count, sum(list(map(len,paths_per_thread))), elapsed_str(time.time()-tic)))

    writer.close()
    output_paths = [ path for thread_paths in output_paths for path  in thread_paths]
    output_labels = [ l for thread_labels in labels_per_thread for l  in thread_labels]
    return output_paths , output_labels

def read_item_list_threaded(paths, mode, storage, id, output_paths):
    if mode == defs.input_mode.image:
        for framepath in paths:
            image = read_image(framepath)
            if image is None:
                return
            storage[id].append(image)
        output_paths[id].extend(paths)
    else:
        for videopath in paths:
            videoframepaths = get_video_frames(videopath)
            vidframes = []
            for framepath in videoframepaths:
                vidframes.append(read_image(framepath))
            if vidframes is None:
                return
            storage[id].extend(vidframes)
            output_paths[id].extend(videoframepaths)

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
def get_video_frames(path):
    frames = []
    if not os.path.exists(path):
        error("Path [%s] does not exist!" % path)

    num_files = len(os.listdir(path))

    # generate a number of frame paths from the video path
    if video_frame_mode == defs.video_frame_mode.random:
        # select frames randomly from the video
        avail_frames = list(range(num_files))
        shuffle(avail_frames)
        avail_frames = avail_frames[:num_frames_per_clip]
        frames.extend(avail_frames)
    elif video_frame_mode == defs.video_frame_mode.clip:
        # get <num_clips> random chunks of a consequtive <num_frames> frames

        possible_chunk_start = list(range(num_files - num_frames_per_clip + 1))
        if len(possible_chunk_start) < num_clips_per_video:
            error("Video %s cannot sustain a number of %d unique %d-frame clips" %
                  (path, num_clips_per_video, num_frames_per_clip))
        shuffle(possible_chunk_start)
        for _ in range(num_clips_per_video):
            start_index = possible_chunk_start[-1]
            possible_chunk_start = possible_chunk_start[:-1]
            clip_frames = list(range(start_index, start_index + num_frames_per_clip))
            frames.extend(clip_frames)

    frame_paths = []
    for fridx in frames:
        basename = os.path.basename(path)
        fr_path = "%s.%04d.%s" % (os.path.join(path,basename),1 + fridx,frame_format)
        frame_paths.append(fr_path)
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
    logger.info("Serializing %s in mode %s" % (inp, mode))
    logger.info("Reading input file.")
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

def write():
    outpaths_per_input = []
    for idx in range(len(input_files)):
        inp = input_files[idx]
        paths, labels, mode = read_file(inp)
        output_file = inp + ".tfrecord"
        logger.info("Writing to %s" % output_file)
        tic = time.time()
        outpaths, outlabels = serialize_multithread(paths, labels,output_file , mode)
        outpaths_per_input.append( ( outpaths, outlabels, mode ) )
        logger.info("Done serializing %s " % inp)
        logger.info("Time elapsed: %s " % elapsed_str(time.time() - tic))
    return outpaths_per_input

# verify the serialization validity
def validate(written_data):

    for index in range(len(input_files)):
        inp = input_files[index]
        paths, labels, mode,  = written_data[index]

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



if len(sys.argv) > 1:
    init_file = sys.argv[-1]

keyvals = initialize_from_file(init_file)

for key in keyvals:
    exec("%s=%s" % (key, keyvals[key]))
print("Successfully initialized from file %s" % init_file)

# outpaths is either the input frame paths in image mode, or the expanded frame paths in video mode
written_data = write()
validate(written_data)
# write the selected clips / frames for video mode
for i in range(len(written_data)):
    inp = input_files[i]
    outfile = inp + ".frames"
    paths, labels, mode = written_data[i]
    if not mode == defs.input_mode.video:
        continue
    logger.info("Documenting selected paths for video mode file %s" % inp)
    logger.info("Writing to file %s" % outfile)
    with open(outfile,"w") as f:
        for path, label in zip(paths,labels):
            f.write("%s %d\n" % (path, label))

