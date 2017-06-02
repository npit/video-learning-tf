import tensorflow as tf
import numpy as np
from random import shuffle
from scipy.misc import imread, imresize, imsave

import logging, time, threading, os
from utils_ import *
import matplotlib.pyplot as plt
import configparser



init_file = "config.ini"

# input paths and folder to prepend to each path in the files
path_prepend_folder = ""
input_files = []
num_threads = 4
num_items_per_thread = 500
raw_image_shape = (240,320,3)
num_frames_per_video = 16
frame_format = "jpg"

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

    if config['path_prepend_folder']:
        path_prepend_folder = config['path_prepend_folder']


    if config['input_files']:
        input_files = eval(config['input_files'])
    if config['num_threads']:
        num_threads = eval(config['num_threads'])
    if config['raw_image_shape']:
        raw_image_shape = eval(config['raw_image_shape'])
    if config['num_items_per_thread']:
        num_items_per_thread = eval(config['num_items_per_thread'])
    if config['frame_format']:
        frame_format = eval(config['frame_format'])
    if config['input_mode']:
        input_mode = eval(config['input_mode'])
    if config['num_frames_per_video']:
        num_frames_per_video = eval(config['num_frames_per_video'])

    print("Successfully initialized from file %s" % init_file)
    return input_files, num_threads, raw_image_shape, num_items_per_thread, frame_format, input_mode, num_frames_per_video


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

    # first of all, write the number of items in the tfrecord
    with open(outfile + ".size","w") as f:
        f.write("%d" % len(paths))

    # split up paths list
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
        for t in range(num_threads_in_run):
            threads[t] = threading.Thread(target=read_item_list_threaded,args=(paths_per_thread[t],mode,frames,t))
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
            if mode == videos:
                duplist = [[label for _ in range(num_frames_per_video)] for label in labels_per_thread[t]]
                labels_per_thread[t] = []
                for l in duplist:
                    labels_per_thread[t].extend(l)

            serialize_to_tfrecord(frames[t], labels_per_thread[t], outfile, writer)
            count += len(frames[t])


        logger.info("Processed %d frames, latest %d-sized batch took %s." %
                    (count, sum(list(map(len,paths_per_thread))), elapsed_str(time.time()-tic)))

    writer.close()




def read_item_list_threaded(paths, mode, storage, id):
    if mode == frames:
        for p in paths:
            image = read_image(p)
            if image is None:
                return
            storage[id].append(image)
    else:
        for p in paths:
            vidframes = get_video_frames(p)
            if vidframes is None:
                return
            storage[id].extend(vidframes)




def serialize_to_tfrecord( frames, labels, outfil, writer):
    for idx in range(len(frames)):
        frame = frames[idx]
        label = labels[idx]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image_shape[0]),
            'width': _int64_feature(image_shape[1]),
            'depth': _int64_feature(image_shape[2]),
            'label': _int64_feature(int(label)),
            'image_raw': _bytes_feature(frame.tostring())}))
        writer.write(example.SerializeToString())



# read all frames for a video
def get_video_frames(path):

    frames = []
    basename = os.path.basename(path)
    for im in range(num_frames_per_video):
        impath = path + "%04d" % (1 + im) + "." + image_format
        image = read_image(impath)
        if image is None:
            return None
        frames.append(image)
    return frames



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
        image = imresize(image, image_shape)

        # there is a problem if we want to store mean-subtracted images, as we'll have to store a float per pixel
        # => 4 x the space of a uint8 image
        # image = image - mean_image
    except Exception as ex:
        logger.error("Error :" + str(ex))
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
            image = img_1d.reshape((image_shape[0], image_shape[1], image_shape[2]))

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
                if path.lower().endswith("." + image_format.lower()):
                    mode = frames
                    logger.info("Set input mode to frames from paths-file items suffixes.")
                else:
                    mode = videos
                    strlen = min(len(path), len(image_format) + 1)
                    suffix = path[-strlen:]
                    logger.info(
                        "Set input mode to videos since paths-file item suffix [%s] differs from image format [%s]." % (
                        suffix, image_format))

            if path_prepend_folder is not None:
                path = os.path.join(path_prepend_folder, path)
            paths.append(path)
            labels.append(int(label.strip()))
    return paths, labels, mode

def write():
    for idx in range(len(input_files)):
        inp = input_files[idx]
        paths, labels, mode = read_file(inp)
        output_file = inp + ".tfrecord"
        logger.info("Writing to %s" % output_file)
        tic = time.time()
        serialize_multithread(paths, labels,output_file , mode)
        logger.info("Done serializing %s " % inp)
        logger.info("Time elapsed: %s " % elapsed_str(time.time() - tic))


def validate():

    for idx in range(len(input_files)):
        inp = input_files[idx]
        paths, labels, mode = read_file(inp)
        num_validate = 10000 if len(paths) >= 10000 else len(paths)
        sound = True
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
            if mode == videos:
                frames = get_video_frames(paths[i])
                fframetf, llabeltf = deserialize_from_tfrecord(iter, num_frames_per_video)
                label = labels[i]

                for v in range(num_frames_per_video):
                    if not np.array_equal(frames[v], fframetf[v]):
                        logger.error("Unequal video frame #%d @ %s" % ( v, paths[i]))
                        sound = False
                    if not label == llabeltf[v]:
                        logger.error("Unequal label at video frame %d @ %s. Found %d, expected %d" % (v, paths[i], label, llabeltf[v]))
                        sound = False


            else:

                frame = read_image(paths[i])
                label = labels[i]

                fframetf, llabeltf = deserialize_from_tfrecord(iter,1)
                frametf = fframetf[0]
                labeltf = llabeltf[0]

                if not np.array_equal(frame , frametf):
                    logger.error("Unequal image @ %s" % paths[i])
                    sound = False
                if not label == labeltf:
                    logger.error("Unequal label @ %s. Found %d, expected %d" % ( paths[i], label, labeltf))
                    sound = False

            lidx = lidx + 1
            if lidx >= len(idx_list):
                break

            testidx = idx_list[lidx]
        if not sound:
            logger.error("errors exist.")
        else:
            logger.info("Validation for %s ok" % (inp + ".tfrecord"))




input_files, num_threads, raw_image_shape, num_items_per_thread, \
frame_format, input_mode, num_frames_per_video = initialize_from_file(init_file)

write()
validate()
