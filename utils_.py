import numpy as np
import tensorflow as tf
import time
import logging
import os


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
def error(msg):

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

# add a tensor to be printed
def add_print_tensor(tens, msg):
    graph_tensor = tf.Print(tens, [tens], message=msg + " " + str(tens.shape))
    return graph_tensor

# view print tf ops
def view_print_tensors(lrcn,dataset, settings,tensorlist):
    if not tensorlist:
        return
    sess = tf.InteractiveSession(); sess.run(tf.global_variables_initializer())
    images, labels_onehot, labels = dataset.read_next_batch(settings)
    sess.run(tensorlist, feed_dict={lrcn.inputData:images, lrcn.inputLabels:labels_onehot})
    sess.close()

# list of sublists of size n from list
def sublist(list, sublist_length):
    return [ list[i:i+sublist_length] for i in range(0, len(list), sublist_length)]

# constants, like C defines. Nesting indicates just convenient hierarchy.
class defs:

    # run phase
    class phase:
        train, val = range(2)
        _str = ["train", "val"]
        def str(arg):
            return defs.phase._str[arg]

    # input mode is framewise dataset vs videowise, each video having n frames
    class input_mode:
        video, image = range(2)
        _str = ["video", "image"]
        def str(arg):
            return defs.input_mode._str[arg]

    # direct reading from disk or from packed tfrecord format
    class data_format:
        raw, tfrecord = range(2)
        _str = ["raw", "tfrecord"]
        def str(arg):
            return defs.data_format._str[arg]

    # run type indicates usage of lstm or singleframe dcnn
    class run_types:
        lstm, singleframe = range(2)
        _str=["lstm","singleframe"]
        def str(arg):
            return defs.run_types._str[arg]

    # batch content type
    images, labels = range(2)
    _str = ["images", "labels"]
    def str(arg):
        return defs.data_format._str[arg]
    class loaded:
        train_index, val_index, epoch_index = range(3)
        _str = ["train_index", "val_index", "epoch_index"]
        def str(arg):
            return defs.loaded._str[arg]