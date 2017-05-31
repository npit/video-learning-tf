
# generic IO
import pickle

# project modules
import lrcn_
import dataset_

# utils
from utils_ import *

import logging



# summaries for training & validation
class Summaries:
    train= []
    val = []
    train_merged = None
    val_merged = None
    def merge(self):
        self.train_merged = tf.summary.merge(self.train)
        self.val_merged = tf.summary.merge(self.val)

    print_tensors = []
# settings class
################
# Generic run settings and parameters should go here
#

class Settings:
    # user - settable parameters
    ################################

    # run mode and type
    run_id = "singleframe_run"
    run_type = defs.run_types.singleframe

    # save / load configuration
    resume_file = "/home/npittaras/single_frame_run/checkpoints/300517_092904-saved_ep_5_btch_2426.graph-2426"
    runFolder = "/home/npittaras/single_frame_run"
    path_prepend_folder = None

    # architecture settings
    lstm_input_layer = "fc7"
    num_classes = 101
    mean_image = [103.939, 116.779, 123.68]

    # data input format
    raw_image_shape = (240, 320, 3)
    image_shape = (227, 227, 3)
    frame_format = defs.data_format.tfrecord
    input_mode = defs.input_mode.image

    # training settings
    do_random_mirroring = True
    do_random_cropping = True
    batch_size_train = 100
    do_training = False#True
    epochs = 15
    optimizer = "SGD"
    learning_rate = 0.001

    # validation settings
    do_validation = True
    validation_interval = 382
    batch_size_val = 88

    # logging
    logging_level = logging.DEBUG
    tensorboard_folder = "tensorboard_graphs"

    # end of user - settable parameters
    ################################

    # internal variables
    ###############################

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
        logfile = os.path.join(self.runFolder ,"log_" +  self.run_id + "_" + get_datetime_str() + ".log")
        print("Using logfile: %s" % logfile)

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

    def set_input_files(self):
        # setup input files
        if self.input_mode == defs.input_mode.image:
            basefilename = "frames"
        elif self.input_mode == defs.input_mode.video:
            basefilename = "videos"

        self.input[defs.phase.train] = os.path.join(self.runFolder, basefilename + ".train")
        self.input[defs.phase.val] = os.path.join(self.runFolder, basefilename + ".test")


    # initialize stuff
    def initialize(self):
        if not os.path.exists(self.runFolder):
            error("Non existent run folder %s" % self.runFolder)

        # configure the logs
        self.configure_logging()

        # if not resuming, set start folder according to now()
        if  self.should_resume():
            # else, do the resume
            self.resume_metadata()

        self.set_input_files()

    # settings are ok
    def good(self):
        if not self.epochs: error("Non positive number of epochs.")
        return True
    # restore dataset meta parameters
    def resume_metadata(self):
        if self.should_resume():
            savefile_metapars = os.path.join(self.runFolder,"checkpoints", self.resume_file + ".snap")

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
            savefile_graph = os.path.join(self.runFolder,"checkpoints", self.resume_file)
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
            checkpoints_folder = os.path.join(self.runFolder, "checkpoints")
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
                [summaries.train_merged, lrcn.loss , lrcn.optimizer],
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
    # validation
    while dataset.loop():
        images, labels_onehot = dataset.read_next_batch()
        dataset.print_iter_info(len(images), len(labels_onehot))
        summaries_val, accuracy = sess.run([summaries.val_merged, lrcn.accuracyVal],
                                           feed_dict={lrcn.inputData: images, lrcn.inputLabels: labels_onehot})
        test_accuracies.append(accuracy)
        accuracy = sum(test_accuracies) / len(test_accuracies)
        dataset.logger.info("Validation run complete, accuracy: %2.5f" % accuracy)
        tboard_writer.add_summary(summaries_val, global_step=dataset.get_global_step())
        tboard_writer.flush()

    return True


# the main function
def main():

    # create and initialize settings and dataset objects
    settings = Settings()
    settings.initialize()

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
