import numpy as np
import tensorflow as tf
import time, configparser, os, logging

# init from config file
def init_config(init_file, tag_to_read):
    if init_file is None:
        return
    if not os.path.exists(init_file):
        error("Unable to read initialization file [%s]." % init_file)
        return

    print("Initializing from file %s" % init_file)
    config = configparser.ConfigParser()
    config.read(init_file)

    if not config[tag_to_read]:
        error('Expected header [%s] in the configuration file!' % tag_to_read)

    config = config[tag_to_read]
    return config

# timestamp print
def elapsed_str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

# datetime for timestamps
def get_datetime_str():
    #return time.strftime("[%d|%m|%y]_[%H:%M:%S]")
    return time.strftime("%d%m%y_%H%M%S")

# error function
def error(msg, logger = None):
    if logger:
        logger.error(msg)
    raise Exception(msg)

# onehot vector generation
def labels_to_one_hot(labels,num_classes):
    onehots = np.zeros(shape=(len(labels),num_classes),dtype=np.int32)

    for l in range(len(labels)):
        onehots[l][labels[l]] = 1

    return onehots

# summary generation
def add_descriptive_summary(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    res = []
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        res.append(tf.summary.scalar('mean', mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        res.append(tf.summary.scalar('stddev', stddev))
        res.append(tf.summary.scalar('max', tf.reduce_max(var)))
        res.append(tf.summary.scalar('min', tf.reduce_min(var)))
        res.append(tf.summary.histogram('histogram', var))
    return res


# list of sublists of size n from list
def sublist(list, sublist_length):
    return [ list[i:i+sublist_length] for i in range(0, len(list), sublist_length)]

# shortcut for tensor printing
def print_tensor(tensor, message, log_level):
    if not log_level == logging.DEBUG:
        return tensor
    tens = tf.Print(tensor,[tensor, tf.shape(tensor)],summarize=10,message=message)
    return tens


def read_file_lines(filename):
    with open(filename, "r") as ff:
        contents = []
        for line in ff:
            contents.append(line.strip())
    return contents
def get_vars_in_scope(starting, scope):
    vars = [v for v in tf.global_variables()]
    for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
        print(i.name)

# constants, like C defines. Nesting indicates just convenient hierarchy.
class defs:

    # run phase
    class phase:
        train, val ="train", "val"

    # input mode is framewise dataset vs videowise, each video having n frames
    class input_mode:
        video, image = "video", "image"

    # direct reading from disk or from packed tfrecord format
    class data_format:
        raw, tfrecord = "raw", "tfrecord"

    # run type indicates usage of lstm or singleframe dcnn
    class workflows:
        lstm, singleframe, imgdesc, videodesc = "lstm","singleframe", "imgdesc", "videodesc"

    # video pooling methods
    class pooling:
        avg, last = "avg", "last"

    # how the video's frames are structured        
    class clipframe_mode:
        rand_frames, rand_clips, iterative = "rand_frames", "rand_clips", "iterative"

    class batch_item:
        default, clip = "default", "clip"

    class optim:
        sgd = "sgd"

    class decay:
        exp, staircase = "exp", "staircase"

    class label_type:
        single, multiple = "single", "multiple"

    class caption_search:
        max = "max"

    class eval_type:
        coco = "coco"

    train_idx, val_idx = 0, 1
    image, label = 0, 1


# trainable object class
class Trainable:
    train_regular = []
    train_modified = []
