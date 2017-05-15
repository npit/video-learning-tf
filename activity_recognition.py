# essential
import math
import os
import argparse

# generic IO
import pickle

# project modules
import lrcn_
import dataset_

# utils
from utils_ import *

import logging
tf.logging.set_verbosity(tf.logging.INFO)


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
    run_id = "run_id"
    networkName = "alexnet"
    epochs = 1

    tensorboard_folder = "/home/nik/uoa/msc-thesis/implementation/tensorboard_graphs"

    verbosity = 1

    # data input format
    frame_format = defs.frame_format.raw
    input_mode = defs.input_mode.image

    # run modes
    BASELINE = "baseline"
    LSTM = "lstm"
    lstm_input_layer = "fc7"

    # optimization method and params
    optimizer = "SGD"
    learning_rate = 0.001

    phase = None

    # test interval
    doTest = True
    testEvery = 50
    saver = tf.train.Saver()
    do_resume = False
    resumeFile = "/home/nik/uoa/msc-thesis/implementation/checkpoints/test_23.04.17_23:31:25"
    saveFolder = "/home/nik/uoa/msc-thesis/implementation/checkpoints"

    # logging
    logging_level = logging.INFO
    log_directory = "logs"
    logger = None
    def configure_logging(self):
        logfile = self.log_directory + os.path.sep + self.run_id + "_" + get_datetime_str() + ".log"
        print("Using logfile: %s" % logfile)
        if not os.path.exists(self.log_directory):
            try:
                os.makedirs(self.log_directory)
            except Exception as ex:
                error(ex)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.logging_level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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


    def initialize(self):
        self.configure_logging()
    # settings are ok
    def good(self):
        if not self.epochs: error("Non positive number of epochs.")
        return True
    # restore dataset meta parameters
    def resume_metadata(self, dataset):
        if self.do_resume:
            savefile_metapars = self.resumeFile + ".snap"
            print2("Resuming iteration snap from file:")
            print2(savefile_metapars, indent=1)
            try:
                # load saved parameters pickle
                with open(savefile_metapars, 'rb') as f:
                    params = pickle.load(f)
                    dataset.load_metaparams(params)
            except Exception as ex:
                error(ex)

    # restore graph variables
    def resume_graph(self, sess):
        if self.do_resume:
            savefile_graph = self.resumeFile + ".graph"
            print2("Resuming run data from file:")
            print2(savefile_graph,indent=1)

            try:
                # load saved graph file
                self.saver.restore(sess, savefile_graph)
            except Exception as ex:
                error(ex)


    # save graph and dataset stuff
    def save(self, sess, dataset,progress, global_step):
        try:
            # save the graph
            now = get_datetime_str()
            basename = self.saveFolder + os.path.sep + self.run_id + "_" + progress + "_" +  now
            savefile_graph = basename + ".graph"
            savefile_metapars = basename + ".snap"

            print2("Saving graph  to [%s]" % savefile_graph, indent=1)
            self.saver.save(sess, savefile_graph, global_step=global_step)
            # save dataset metaparams
            print2("Saving params to [%s]" % savefile_metapars, indent=1)
            params2save = []
            params2save.extend([ dataset.batchIndexTrain, dataset.batchIndexVal, dataset.epochIndex])

            with open(savefile_metapars,'wb') as f:
                params = [dataset.batchIndexTrain, dataset.batchIndexVal, dataset.outputFolder]
                pickle.dump(params,f)
        except Exception as ex:
            error(ex)


# train the network
def train_test(settings, dataset, lrcn, sess, tboard_writer, summaries):
    dataset.set_phase(defs.phase.train)
    if not settings.good():
        error("Wacky configuration, exiting.")

    for epochIdx in range(dataset.epochs):
        print2('Epoch %d / %d ' % (1 + epochIdx, dataset.epochs), pr_type="banner==")
        settings.logger.info('Epoch %d / %d ' % (1 + epochIdx, dataset.epochs))

        while dataset.loop() and False:
            # read  batch
            images, labels_onehot = dataset.read_next_batch()
            dataset.print_iter_info( len(images) , len(labels_onehot))
            summaries_train, batch_loss, _, accuracy = sess.run([summaries.train_merged, lrcn.loss , lrcn.optimizer, lrcn.accuracyTrain], feed_dict={lrcn.inputData:images, lrcn.inputLabels:labels_onehot})
            print("Train accuracy: %2.5f" % accuracy)
            settings.logger.info("Train accuracy: %2.5f" % accuracy)

            tboard_writer.add_summary(summaries_train, global_step=dataset.get_global_step())
            tboard_writer.flush()

            if settings.doTest:
                test(dataset, lrcn, sess, tboard_writer, summaries)

        # save a checkpoint every epoch
        settings.save(sess, dataset, progress="ep_%d_btch_%d" % (1+epochIdx, dataset.get_global_step()),
                          global_step=dataset.get_global_step())
        dataset.epochIndex = dataset.epochIndex + 1

# tests the network on validation data
def test(dataset, lrcn, sess, tboard_writer, summaries):



    if dataset.should_test_now():
        test_accuracies = []
        dataset.set_phase(defs.phase.val)
        # validation
        while dataset.loop():
            images, labels_onehot = dataset.read_next_batch()
            dataset.print_iter_info(len(images), len(labels_onehot))

            summaries_val, accuracy = sess.run([summaries.val_merged, lrcn.accuracyVal],
                                               feed_dict={lrcn.inputData: images, lrcn.inputLabels: labels_onehot})
            test_accuracies.append(accuracy)
        accuracy = sum(test_accuracies) / len(test_accuracies)
        print("Validation accuracy: %2.5f" % accuracy)
        tboard_writer.add_summary(summaries_val, global_step=dataset.get_global_step())
        tboard_writer.flush()

# the main function
def main():


    # argparse is quite overkill, but future proof
    parser = argparse.ArgumentParser(description="Run the activity recognition task.")
    parser.add_argument("run_mode", metavar='mode', type=str,
                    help='an integer for the accumulator')
    args = parser.parse_args()






    # create and initialize settings and dataset objects
    settings = Settings()
    settings.initialize()

    settings.logger.info('Running the activity recognition task in mode: [%s]' % args.run_mode)
    # init summaries for printage
    summaries=Summaries()

    print2('Running the activity recognition task in mode: [%s]' % args.run_mode, pr_type="banner##")
    print() # newline from Lidl

    dataset = dataset_.Dataset()
    # resume iteration, batchsize, etc
    settings.resume_metadata(dataset)
    # initialize & pre-process dataset; checks and respects resumed vars
    dataset.initialize(settings)

    # create and configure the nets : CNN and / or lstm
    lrcn = lrcn_. LRCN()
    lrcn.create(settings, dataset, args.run_mode, summaries)

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

    train_test(settings, dataset, lrcn,  sess, tboard_writer, summaries)

    # mop up
    tboard_writer.close()
    sess.close()

if __name__ == "__main__":
    main()
