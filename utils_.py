import numpy as np
import tensorflow as tf
import time, configparser, os, logging, sys, re, math

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
def elapsed_str(previous_tic, up_to = None):
    if up_to is None:
        up_to = time.time()
    duration_sec = up_to - previous_tic
    m, s = divmod(duration_sec, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

# datetime for timestamps
def get_datetime_str():
    #return time.strftime("[%d|%m|%y]_[%H:%M:%S]")
    return time.strftime("%d%m%y_%H%M%S")

# logging setup
class CustomLogger:
    loggername='default'
    logging_level = logging.INFO
    instance = None
    # have a container to store log messages to be displayed with a delay
    log_storage = {}
    def get_log_storage(self, storage_id):
        if storage_id in self.log_storage:
            return self.log_storage[storage_id]
        return []
    def clear_log_storage(self, storage_id):
        del self.log_storage[storage_id]

    def add_to_log_storage(self, storage_id,message):
        if storage_id not in self.log_storage:
            self.log_storage[storage_id] = []
        self.log_storage[storage_id].append(message)




    # configure logging settings
    def configure_logging(self, logfile, logging_level):
        print("Initializing logging to logfile: %s" % logfile)
        sys.stdout.flush()

        self.logging_level = logging_level
        self.logger = logging.getLogger('default')
        self.logger.setLevel(self.logging_level)

        formatter = logging.Formatter('%(asctime)s| %(levelname)7s - %(filename)15s - line %(lineno)4d - %(message)s')

        # file handler
        handler = logging.FileHandler(logfile)
        handler.setLevel(self.logging_level)
        handler.setFormatter(formatter)
        # console handler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(handler)
        self.logger.addHandler(consoleHandler)
    def get_logging_level():
        return CustomLogger.instance.logger.level

# shortcut log functions
def error(msg):
    error_(msg)
    raise Exception(msg)
def info(message):
    logging.getLogger(CustomLogger.loggername).info(message)
def warning(message):
    logging.getLogger(CustomLogger.loggername).warning(message)
def error_( message):
    logging.getLogger(CustomLogger.loggername).error(message)
def debug(message):
    logging.getLogger(CustomLogger.loggername).debug(message)

# onehot vector generation
def labels_to_one_hot(labels,num_classes):
    if not type(labels) == list:
        labels= [labels]
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
def sublist(llist, sublist_length, only_num = False):
    divisions = range(0, len(llist), sublist_length)
    if only_num:
        # just the lengths
        return [len(d) for d in divisions]
    return [ llist[i:i+sublist_length] for i in divisions]

# shortcut for tensor printing
def print_tensor(tensor, message):
    if not CustomLogger.get_logging_level() == logging.DEBUG:
        return tensor
    tensor_cols = 20
    tens = tf.Print(tensor,[tensor, tf.shape(tensor)],summarize=2*tensor_cols ,message=message)
    return tens

# duplicate elements from list
def duplicates(llist):
    return set([x for x in llist if llist.count(x) > 1])

# read lines from text file, cleaning whitespace
def read_file_lines(filename):
    with open(filename, "r") as ff:
        contents = []
        for line in ff:
            contents.append(line.strip())
    return contents

# read dictionary-line lines from text file, in the format key\tvalue
def read_file_dict(filename):
    dictionary = {}
    with open(filename,"r") as f:
        for line in f:
            key, value = line.strip().split("\t")
            key, value = key.strip(),value.strip()
            if key in dictionary:
                logging.getLogger().warning("Duplicate key %s in file %s" % (key, filename))
            dictionary[key] = value
    return dictionary

# remove trailing name index
def drop_tensor_name_index(name):
    return ":".join(name.split(":")[0:-1])

# trainable object class
class Trainable:
    train_regular = []
    train_modified = []
    ignorable_variable_names = []
    def __init__(self):
        self.train_regular = []
        self.train_modified = []
        self.ignorable_variable_names = []



