
# generic IO
import pickle
import sys

# project modules
import lrcn_
import dataset_

# utils
from utils_ import *

import logging, configparser



# summaries for training & validation
class Summaries:
    train= []
    val = []
    train_merged = None
    val_merged = None
    def merge(self):
        self.train_merged = tf.summary.merge(self.train)
        self.val_merged = tf.summary.merge(self.val)

# settings class
################
# Generic run settings and parameters should go here
#

class Settings:
    # user - settable parameters
    ################################
    # initialization file
    init_file = "config.ini"

    # run mode and type
    run_id = "run_id"
    run_type = defs.run_types.singleframe

    # save / load configuration
    resume_file = None
    run_folder = None
    path_prepend_folder = None

    # architecture settings
    lstm_input_layer = "fc7"
    lstm_num_hidden = 256
    video_pooling_type = defs.pooling.avg
    num_classes = None
    mean_image = None

    # data input format
    raw_image_shape = (240, 320, 3)
    image_shape = (227, 227, 3)
    data_format = defs.data_format.raw
    frame_format = "jpg"
    input_mode = defs.input_mode.image
    num_frames_per_clip = 16
    num_clips_per_video = 1

    # training settings
    do_random_mirroring = True
    do_random_cropping = True
    batch_size_train = 100
    do_training = False
    epochs = 15
    optimizer = "SGD"
    learning_rate = 0.001
    dropout_keep_prob = 0.5

    # validation settings
    do_validation = True
    validation_interval = 1     # train iterations interval for validation or percentage
    batch_size_val = 88

    # logging
    logging_level = logging.DEBUG
    tensorboard_folder = "tensorboard_graphs"

    # end of user - settable parameters
    ################################

    # internal variables
    ###############################

    # initialization


    # input data files
    input = [[],[]]

    # training settings
    epoch_index = 0
    train_index = 0

    # validation settings
    val_index = 0

    # logging
    logger = None

    # misc
    saver = None

    # should resume
    def should_resume(self):
        return self.resume_file is not None

    # configure logging settings
    def configure_logging(self):
        #tf.logging.set_verbosity(tf.logging.INFO)
        logfile = os.path.join(self.run_folder ,"log_" +  self.run_id + "_" + get_datetime_str() + ".log")
        print("Using logfile: %s" % logfile)
        sys.stdout.flush()
        self.logger = logging.getLogger(__name__)
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

    # set the input files
    def set_input_files(self):
        # setup input files
        if self.input_mode == defs.input_mode.image:
            basefilename = "frames"
        elif self.input_mode == defs.input_mode.video:
            basefilename = "videos"
        else:
            self.logger.error("Undefined input mode: [%s]" % str(self.input_mode))
            error("Undefined input mode")
        self.input[defs.phase.train] = os.path.join(self.run_folder, basefilename + ".train")
        self.input[defs.phase.val] = os.path.join(self.run_folder, basefilename + ".test")

    # file initialization
    def initialize_from_file(self):
        if self.init_file is None:
            return
        if not os.path.exists(self.init_file):
            error("Unable to read initialization file [%s]." % self.init_file)
            return

        tag_to_read = "run"

        print("Initializing from file %s" % self.init_file)
        config = configparser.ConfigParser()
        config.read(self.init_file)

        if not config[tag_to_read]:
            error('Expected header [%s] in the configuration file!' % tag_to_read)

        config = config[tag_to_read]

        if config['run_id']:
            self.run_id = eval(config['run_id'])
        if config['run_type']:
            self.run_type = eval(config['run_type'])
        if config['resume_file']:
            self.resume_file = eval(config['resume_file'])
        if config['run_folder']:
            self.run_folder = eval(config['run_folder'])
        if config['path_prepend_folder']:
            self.path_prepend_folder = eval(config['path_prepend_folder'])
        if config['lstm_input_layer']:
            self.lstm_input_layer = eval(config['lstm_input_layer'])
        if config['num_classes']:
            self.num_classes = eval(config['num_classes'])
        if config['mean_image']:
            self.mean_image = eval(config['mean_image'])
        if config['raw_image_shape']:
            self.raw_image_shape = eval(config['raw_image_shape'])
        if config['image_shape']:
            self.image_shape = eval(config['image_shape'])
        if config['frame_format']:
            self.frame_format = eval(config['frame_format'])
        if config['data_format']:
            self.data_format = eval(config['data_format'])
        if config['input_mode']:
            self.input_mode = eval(config['input_mode'])
        if config['do_random_mirroring']:
            self.do_random_mirroring = eval(config['do_random_mirroring'])
        if config['do_random_cropping']:
            self.do_random_cropping = eval(config['do_random_cropping'])
        if config['batch_size_train']:
            self.batch_size_train = eval(config['batch_size_train'])
        if config['do_training']:
            self.do_training = eval(config['do_training'])
        if config['epochs']:
            self.epochs = eval(config['epochs'])
        if config['optimizer']:
            self.optimizer = eval(config['optimizer'])
        if config['learning_rate']:
            self.learning_rate = eval(config['learning_rate'])
        if config['do_validation']:
            self.do_validation = eval(config['do_validation'])
        if config['validation_interval']:
            self.validation_interval = eval(config['validation_interval'])
        if config['batch_size_val']:
            self.batch_size_val = eval(config['batch_size_val'])
        if config['logging_level']:
            self.logging_level = eval(config['logging_level'])
        if config['tensorboard_folder']:
            self.tensorboard_folder = eval(config['tensorboard_folder'])
        print("Successfully initialized from file %s" % self.init_file)

    # initialize stuff
    def initialize(self, args):
        if len(args) > 1:
            self.init_file = args[-1]

        self.initialize_from_file()
        if not os.path.exists(self.run_folder):
            error("Non existent run folder %s" % self.run_folder)
        if not self.do_validation and not self.do_training:
            error("Neither training nor validation is enabled.")
        # configure the logs
        self.configure_logging()

        # if not resuming, set start folder according to now()
        if  self.should_resume():
            if self.do_training:
                self.logger.info("Resuming training.")
            # else, do the resume
            self.resume_metadata()
        else:
            if self.do_training:
                self.logger.info("Starting training from scratch.")
        self.set_input_files()

    # settings are ok
    def good(self):
        if not self.epochs: error("Non positive number of epochs.")
        return True

    # restore dataset meta parameters
    def resume_metadata(self):
        if self.should_resume():
            savefile_metapars = os.path.join(self.run_folder,"checkpoints", self.resume_file + ".snap")

            self.logger.info("Resuming metadata from file:" + savefile_metapars)

            try:
                # load saved parameters pickle
                with open(savefile_metapars, 'rb') as f:
                    params = pickle.load(f)
            except Exception as ex:
                error(ex)

            # set run options from loaded stuff
            self.train_index = params[defs.loaded.train_index]
            self.val_index = params[defs.loaded.val_index]
            self.epoch_index = params[defs.loaded.epoch_index]
            self.logger.info("Restored epoch %d, train index %d, validation index %d" % (self.epoch_index, self.train_index, self.val_index))

    # restore graph variables
    def resume_graph(self, sess):
        if self.should_resume():
            if self.saver is None:
                self.saver = tf.train.Saver()
            savefile_graph = os.path.join(self.run_folder,"checkpoints", self.resume_file)
            self.logger.info("Resuming tf graph from file:" + savefile_graph)

            try:
                # load saved graph file
                self.saver.restore(sess, savefile_graph)
            except Exception as ex:
                error(ex)

    # save graph and dataset stuff
    def save(self, sess, dataset,progress, global_step):
        try:
            if self.saver is None:
                self.saver = tf.train.Saver()
            # save the graph
            checkpoints_folder = os.path.join(self.run_folder, "checkpoints")
            if not os.path.exists(checkpoints_folder):
                os.makedirs(checkpoints_folder)

            basename = os.path.join(checkpoints_folder,get_datetime_str() + "-saved_" + progress)
            savefile_graph = basename + ".graph"

            self.logger.info("Saving graph  to [%s]" % savefile_graph)
            saved_instance_name = self.saver.save(sess, savefile_graph, global_step=global_step)

            # save dataset metaparams
            savefile_metapars = saved_instance_name + ".snap"

            self.logger.info("Saving params to [%s]" % savefile_metapars)
            self.logger.info("Saving params epoch %d, train index %d, validation index %d" %
                (dataset.epoch_index, dataset.batch_index_train, dataset.batch_index_val))

            params2save = [[],[],[]]
            params2save[defs.loaded.train_index] = dataset.batch_index_train
            params2save[defs.loaded.val_index] = dataset.batch_index_val
            params2save[defs.loaded.epoch_index] = dataset.epoch_index

            with open(savefile_metapars,'wb') as f:
                pickle.dump(params2save,f)
        except Exception as ex:
            error(ex)


# train the network
def train_test(settings, dataset, lrcn, sess, tboard_writer, summaries):
    settings.logger.info("Starting train/test")
    start_time = time.time()
    timings = []


    if not settings.good():
        error("Wacky configuration, exiting.")

    for epochIdx in range(dataset.epochs):
        dataset.set_or_swap_phase(defs.phase.train)
        while dataset.loop():
            # read  batch
            images, labels_onehot = dataset.read_next_batch()
            dataset.print_iter_info( len(images) , len(labels_onehot))

            summaries_train, batch_loss, _ = sess.run(
                [summaries.train_merged, lrcn.loss, lrcn.optimizer],
                feed_dict={lrcn.inputData:images, lrcn.inputLabels:labels_onehot})

            settings.logger.info("Batch loss : %2.5f" % batch_loss)

            tboard_writer.add_summary(summaries_train, global_step=dataset.get_global_step())
            tboard_writer.flush()

            if settings.do_validation:
                test_ran = test(dataset, lrcn, sess, tboard_writer, summaries)
                if test_ran:
                    dataset.set_or_swap_phase(defs.phase.train)

        dataset.logger.info("Epoch [%d] training run complete." % (1+epochIdx))
        # save a checkpoint every epoch
        settings.save(sess, dataset, progress="ep_%d_btch_%d" % (1+epochIdx, dataset.get_global_step()),
                          global_step=dataset.get_global_step())
        dataset.epoch_index = dataset.epoch_index + 1
        timings.append(time.time() - start_time)
        dataset.reset_phase(defs.phase.train)
        settings.logger.info("Time elapsed for epoch %d : %s ." % (1+epochIdx, elapsed_str(timings[epochIdx])))

    settings.logger.info("Time elapsed for %d epochs : %s ." % (settings.epochs, elapsed_str(sum(timings))))

# test the network on validation data
def test(dataset, lrcn, sess, tboard_writer, summaries):
    # if testing within training, check if it should run
    if dataset.do_training:
        if not dataset.should_test_now():
            return False
        else:
            dataset.set_or_swap_phase(defs.phase.val)
    else:
        dataset.set_current_phase(defs.phase.val)

    # reset validation phase indexes
    dataset.reset_phase(dataset.phase)

    test_accuracies = []
    test_logits = []
    test_labels = []
    dataset.logger.debug("Gathering test logits")
    # validation loop
    while dataset.loop():
        # get images and labels
        images, labels_onehot = dataset.read_next_batch()
        dataset.print_iter_info(len(images), len(labels_onehot))
        summaries_val, logits = sess.run([summaries.val_merged, lrcn.accuracyVal],
                                           feed_dict={lrcn.inputData: images, lrcn.inputLabels: labels_onehot})
        test_logits.append(logits)
        test_labels.append(labels_onehot)
    # compute accuracy
    dataset.logger.debug("Computing test accuracy")
    if dataset.num_clips_per_video > 1:
        # for multiple clips, we have to do an extra inter-clip aggregation per video
        # an important assumption is that clips and clipframes are sequential in the dataset
        # TODO: for better security, add a video index + clip index output files @ serialization script
        # group data per video
        num_frames_per_video = dataset.num_clips_per_video * dataset.num_frames_per_clip
        test_logits = sublist(test_logits, num_frames_per_video)
        test_labels = sublist(test_labels, num_frames_per_video)
        # aggregate
        for vid_idx in range(len(test_logits)):
            video_logits = test_logits[vid_idx]
            # aggregate the frames per clip, ending in num_clips logit vectors
             = np.mean()


    else:
        # no clip-level aggregation necessary. Go straight from frames to video
    accuracy = sum(test_accuracies) / len(test_accuracies)
    dataset.logger.info("Validation run complete, accuracy: %2.5f" % accuracy)
    tboard_writer.add_summary(summaries_val, global_step=dataset.get_global_step())
    tboard_writer.flush()

    return True


# the main function
def main():

    # create and initialize settings and dataset objects
    settings = Settings()
    settings.initialize(sys.argv)

    settings.logger.info('Running the activity recognition task in mode: [%s]' % defs.run_types.str(settings.run_type))
    # init summaries for printage
    summaries=Summaries()

    # initialize & pre-process dataset
    dataset = dataset_.Dataset()
    dataset.initialize(settings)

    # create and configure the nets : CNN and / or lstm
    lrcn = lrcn_. LRCN()
    lrcn.create(settings, dataset, settings.run_type, summaries)

    # view_print_tensors(lrcn,dataset,settings,lrcn.print_tensors)

    # create and init. session and visualization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # restore graph variables,
    settings.resume_graph(sess)

    # mop up all summaries. Unless you separate val and train, the training subgraph
    # will be executed when calling the summaries op. Which is bad.
    summaries.merge()

    # create the writer for visualizashuns
    tboard_writer = tf.summary.FileWriter(settings.tensorboard_folder, sess.graph)
    if settings.do_training:
        train_test(settings, dataset, lrcn,  sess, tboard_writer, summaries)
    elif settings.do_validation:
        test(dataset, lrcn,  sess, tboard_writer, summaries)

    # mop up
    tboard_writer.close()
    sess.close()
    settings.logger.info("Run [%s] complete." % settings.run_id)

if __name__ == "__main__":
    main()
