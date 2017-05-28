import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave

import logging, time, threading, os
from utils_ import *
#import matplotlib.pyplot as plt

# frames or videos
frames, videos = range(2)

# input paths and folder to prepend to each path in the files
path_prepend_folder = "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames"
input_files = [
"/home/nik/uoa/msc-thesis/implementation/examples/test_run/frames.train",
"/home/nik/uoa/msc-thesis/implementation/examples/test_run/frames.test"
]
num_threads = 4
num_items_per_thread = 500

image_shape = (240,320,3)

mean_image = [103.939, 116.779, 123.68]
height = image_shape[0]
width = image_shape[1]
blue = np.full((height, width), mean_image[0])
green = np.full((height, width), mean_image[1])
red = np.full((height, width), mean_image[2])
mean_image = np.stack([blue, green, red])
mean_image = np.transpose(mean_image,[1,2,0])
mean_image = np.ndarray.astype(mean_image,np.uint8)

num_frames_per_video = 16
image_format = "jpg"

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

    if mode == videos:
        duplist = [ [ label for j in range(num_frames_per_video) ] for label in labels ]
        labels = []
        for l in duplist:
            labels.extend(l)

    count = 0
    writer = tf.python_io.TFRecordWriter(outfile)
    for paths_in_run in paths_per_thread_run:
        tic = time.time()
        logger.debug("Processing %d items for the run." % len(paths_in_run))
        paths_per_thread = sublist(paths_in_run, num_items_per_thread )
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
            serialize_to_tfrecord(frames[t], labels, outfile, writer)
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


# def display_image(image,label=None):
#     print(label)
#     plt.title(label)
#     plt.imshow(image)
#     plt.show()
#     # plt.waitforbuttonpress()


def deserialize_example(ex):
    features = tf.parse_single_example(
        ex,
        # Defaults are not specified since both keys are required.
        features={
            'height' : tf.FixedLenFeature([],tf.int64)
            ,'width': tf.FixedLenFeature([], tf.int64)
            ,'depth': tf.FixedLenFeature([], tf.int64)
            ,'image_raw': tf.FixedLenFeature([], tf.string)
            #, 'label': tf.FixedLenFeature([], tf.int64)
        })
    return tf.decode_raw(features['image_raw'], tf.uint8)


def write():
    for idx in range(len(input_files)):

        mode = None
        inp = input_files[idx]
        logger.info("Serializing %s in mode %s" % (inp,mode))
        logger.info("Reading input file.")
        paths = []
        labels = []
        with open(inp,'r') as f:
            for line in f:

                path, label = line.split(' ')
                path = path.strip()

                if mode is None:
                    if path.lower().endswith("." + image_format.lower()):
                        mode = frames
                        logger.info("Set input mode to frames from paths-file items suffixes.")
                    else:
                        mode = videos
                        strlen = min(len(path), len(image_format)  + 1)
                        suffix = path[-strlen:]
                        logger.info("Set input mode to videos since paths-file item suffix [%s] differs from image format [%s]." % (suffix, image_format))

                if path_prepend_folder is not None:
                    path = os.path.join(path_prepend_folder, path)
                paths.append(path)
                labels.append(int(label.strip()))

        output_file = inp + ".tfrecord"
        logger.info("Writing to %s" % output_file)
        tic = time.time()
        serialize_multithread(paths, labels,output_file , mode)
        logger.info("Done serializing %s " % inp)
        logger.info("Time elapsed: %s " % elapsed_str(time.time() - tic))



write()
