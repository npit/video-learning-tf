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
def elapsed_str(seconds):
    m, s = divmod(seconds, 60)
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
def print_tensor(tensor, message):
    if not CustomLogger.get_logging_level() == logging.DEBUG:
        return tensor
    tensor_cols = 20
    tens = tf.Print(tensor,[tensor, tf.shape(tensor)],summarize=2*tensor_cols ,message=message)
    return tens

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


# constants, like C defines. Nesting indicates just convenient hierarchy.
class defs:

    # run phase
    class phase:
        train, val ="train", "val"

    # input mode is framewise dataset vs videowise, each video having n frames
    class input_mode:
        video, image = "video", "image"
        def get_from_workflow(arg):
            if defs.workflows.is_image(arg):
                return defs.input_mode.image
            elif defs.workflows.is_video(arg):
                return defs.input_mode.video
            else:
                error("No input mode discernible from workflow %s" % arg)
                return None

    # direct reading from disk or from packed tfrecord format
    class data_format:
        raw, tfrecord = "raw", "tfrecord"
    class rnn_visual_mode:
        state_bias, input_bias, input_concat = "state_bias", "input_bias", "input_concat"
    # run type indicates usage of lstm or singleframe dcnn
    class workflows:
        class acrec:
            singleframe, lstm = "acrec_singleframe", "acrec_lstm"
            def is_workflow(arg):
                return arg == defs.workflows.acrec.singleframe or \
                       arg == defs.workflows.acrec.lstm
        class imgdesc:
            statebias, inputstep = "imgdesc_statebias", "imgdesc_inputstep"
            def is_workflow(arg):
                return arg == defs.workflows.imgdesc.statebias or \
                       arg == defs.workflows.imgdesc.inputstep
        class videodesc:
            pooled, encdec = "videodesc_pooled", "videodesc_encdec"
            def is_workflow(arg):
                return arg == defs.workflows.videodesc.pooled or \
                       arg == defs.workflows.videodesc.encdec
        def is_description(arg):
            return defs.workflows.imgdesc.is_workflow(arg) or \
                    defs.workflows.videodesc.is_workflow(arg)
        def is_video(arg):
            return defs.workflows.acrec.singleframe == arg or \
                   defs.workflows.acrec.lstm== arg or \
                   defs.workflows.videodesc.encdec == arg or \
                   defs.workflows.videodesc.pooled == arg
        def is_image(arg):
            return defs.workflows.imgdesc.statebias == arg or \
                   defs.workflows.imgdesc.inputstep == arg

        #lstm, singleframe, imgdesc, videodesc = "lstm","singleframe", "imgdesc", "videodesc"

    # video pooling methods
    class pooling:
        avg, last, reshape = "avg", "last", "reshape"

    # how the video's frames are structured        
    class clipframe_mode:
        rand_frames, rand_clips, iterative = "rand_frames", "rand_clips", "iterative"

    class batch_item:
        default, clip = "default", "clip"

    class optim:
        sgd, adam = "sgd", "adam"

    # learning rate decay parameters
    class decay:
        # granularity level
        class granularity:
            exp, staircase = "exp", "staircase"
        # drop at intervals or a total number of times
        class scheme:
            interval, total = "interval", "total"

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
    def __init__(self):
        self.train_regular = []
        self.train_modified = []

# eye candy
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')
        self.current = self.current + 1


    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)