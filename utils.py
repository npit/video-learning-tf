import numpy as np
import tensorflow as tf
import time

# datetime for timestamps
def get_datetime_str():
    return time.strftime("%d.%m.%y_%H:%M:%S")
# error function
def error(msg):

    raise Exception(msg)

def labels_to_one_hot(labels,num_classes):
    onehots = np.zeros(shape=(len(labels),num_classes),dtype=np.int32)

    for l in range(len(labels)):
        onehots[l][labels[l]] = 1
    return onehots

def print2(msg, indent=0, type="", verbose = True):
    if not verbose:
        return
    ind = ''
    for i in range(indent):
        ind+= '\t'
    bann = ""
    if type == "banner":
        for _ in msg:
            bann+='#'
        print(ind + bann)
    print (ind+msg)
    if type == "banner":
        print(ind + bann)

def add_descriptive_summary(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
