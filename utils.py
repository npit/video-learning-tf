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

def print2(msg, indent=0, pr_type="", req_lvl = 0, lvl = 0):
    # print according to verbosity level
    if req_lvl > lvl:
        return
    ind = ''
    for i in range(indent):
        ind+= '\t'

    banner = ""
    banner_tok = "#"

    if pr_type.startswith("banner"):
        if len(pr_type) > 6:
            banner_tok = pr_type[6:]
        for _ in msg:
            banner+=banner_tok
        print(ind + banner)
    print (ind+msg)
    if pr_type.startswith("banner"):
        print(ind + banner)

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