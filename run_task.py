
# generic IO
import pickle
import sys

# project modules
import lrcn_
import dataset_

# utils
from utils_ import *

import logging, configparser

import json

# summaries for training & validation
class Summaries:
    train= []
    val = []
    train_merged = None
    val_merged = None
    def merge(self):
        if self.train:
            self.train_merged = tf.summary.merge(self.train)
        if self.val:
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
    run_id = ""
    workflow = defs.workflows.singleframe

    # save / load configuration
    resume_file = None
    run_folder = None
    path_prepend_folder = None

    # architecture settings
    frame_encoding_layer = "fc7"
    lstm_num_hidden = 256
    frame_pooling_type = defs.pooling.avg
    clip_pooling_type = defs.pooling.avg
    num_classes = None
    mean_image = None

    # data input format
    raw_image_shape = (240, 320, 3)
    image_shape = (227, 227, 3)
    data_format = defs.data_format.tfrecord
    frame_format = "jpg"
    input_mode = defs.input_mode.image
    num_frames_per_clip = 16
    clips_per_video = 1

    # training settings
    do_random_mirroring = True
    do_random_cropping = True
    batch_size_train = 100
    do_training = False
    epochs = 15
    optimizer = defs.optim.sgd
    base_lr = 0.001
    lr_mult = 2
    lr_decay = (defs.decay.exp, 1000, 0.96)
    dropout_keep_prob = 0.5

    # validation settings
    do_validation = True
    validation_interval = 1
    batch_size_val = 88
    batch_item = defs.batch_item.default

    # caption generation settings
    caption_search = defs.caption_search.max
    eval_type = defs.eval_type.coco

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

    # logging
    logger = None

    # misc
    saver = None

    # should resume, if set and non-empty
    def should_resume(self):
        return self.resume_file is not None and self.resume_file

    # configure logging settings
    def configure_logging(self):
        #tf.logging.set_verbosity(tf.logging.INFO)
        logfile = os.path.join(self.run_folder ,"log_" +  self.run_id + "_" + get_datetime_str() + ".log")
        print("Initializing logging to logfile: %s" % logfile)
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
        basefilename = "data"
        self.input[defs.train_idx] = os.path.join(self.run_folder, basefilename + ".train")
        self.input[defs.val_idx] = os.path.join(self.run_folder, basefilename + ".test")

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

        for var in config:
            exec("self.%s=%s" % (var, config[var]))
        # set append the config.ini.xxx suffix to the run id
        trainval = ""
        if self.do_training:
            trainval = "train"
        if self.do_validation:
            trainval = trainval + "val"
        if self.should_resume():
            trainval = trainval + "_resume"
        else:
            trainval = trainval + "_scratch"
        if self.run_id:
            self.run_id = "_" + self.run_id
        self.run_id = self.workflow + self.run_id + "_" + trainval
        if not self.init_file == "config.ini":
            init_file_suffix = self.init_file.split(".")
            self.run_id = self.run_id + "_" + init_file_suffix[-1]
        print("Initialized run [%s] from file %s" % ( self.run_id, self.init_file))
        sys.stdout.flush()

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
        sys.stdout.flush(), sys.stderr.flush()
        self.logger.info("Starting [%s] workflow." % self.workflow)
        # if not resuming, set start folder according to now()
        if  self.should_resume():
            if self.do_training:
                # load batch and epoch where training left off
                self.logger.info("Resuming training.")
                # resume training metadata only in training
                self.resume_metadata()
        else:
            if self.do_training:
                self.logger.info("Starting training from scratch.")
            if not self.do_training and self.do_validation:
                self.logger.warning("Starting validation-only run with an untrained network.")
        self.set_input_files()
        if not self.good():
            error("Wacky configuration, exiting.")

    # check if settings are ok
    def good(self):
        if not self.epochs: error("Non positive number of epochs.")
        if self.batch_item == defs.batch_item.clip and self.input_mode == defs.input_mode.image:
            error("Clip batches set with image input mode")
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
            self.train_index, self.epoch_index = params
            self.logger.info("Restored epoch %d, train index %d" % (self.epoch_index, self.train_index))

    # restore graph variables
    def resume_graph(self, sess):
        if self.should_resume():
            if self.saver is None:
                self.saver = tf.train.Saver()
            savefile_graph = os.path.join(self.run_folder,"checkpoints", self.resume_file)
            self.logger.info("Resuming tf graph from file:" + savefile_graph)

            try:
                # load saved graph file
                tf.reset_default_graph()
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

            basename = os.path.join(checkpoints_folder, get_datetime_str() + "_" + self.workflow + "_" + progress)
            savefile_graph = basename + ".graph"

            self.logger.info("Saving graph  to [%s]" % savefile_graph)
            saved_instance_name = self.saver.save(sess, savefile_graph, global_step=global_step)

            # save dataset metaparams
            savefile_metapars = saved_instance_name + ".snap"

            self.logger.info("Saving params to [%s]" % savefile_metapars)
            self.logger.info("Saving params epoch %d, train index %d" %
                (dataset.epoch_index, dataset.batch_index_train))

            params2save = [dataset.batch_index_train , dataset.epoch_index]

            with open(savefile_metapars,'wb') as f:
                pickle.dump(params2save,f)
        except Exception as ex:
            error(ex)


def get_feed_dict(lrcn, settings, images, ground_truth):
    fdict = {}
    fdict[lrcn.inputData] = images


    if settings.workflow == defs.workflows.imgdesc:
        # get words per caption, onehot labels, embeddings

        embeddings = ground_truth[0]
        labels = ground_truth[1]
        metadata = ground_truth[2]
        fdict[lrcn.inputLabels] = labels
        fdict[lrcn.words_per_image] = metadata
        fdict[lrcn.word_embeddings] = embeddings
        # create base index list
        idx_list = []
        seq_len =  int(len(embeddings) / 2)
        # for the batch item other than the first, add the offset of the sequence length
        for idx, num_captions in enumerate(metadata):
            offset = idx * seq_len
            # generate, adding one more index for the EOS
            idxs_of_seq = [i + offset for i in range(num_captions + 1)]


            idx_list.extend(idxs_of_seq)
        binary_word_idx = np.asarray(idx_list,np.int32)
        fdict[lrcn.binary_word_idx] = binary_word_idx

        num_labels = len(labels)
    else:
        fdict[lrcn.inputLabels] = ground_truth
        num_labels = len(ground_truth)

    return fdict, num_labels





# train the network
def train_test(settings, dataset, lrcn, sess, tboard_writer, summaries):
    settings.logger.info("Starting train/test")
    run_batch_count = 0

    for epochIdx in range(dataset.epoch_index, dataset.epochs):
        dataset.set_or_swap_phase(defs.phase.train)
        while dataset.loop():
            # read  batch
            images, ground_truth = dataset.read_next_batch()
            fdict, num_labels = get_feed_dict(lrcn, settings, images, ground_truth)
            dataset.print_iter_info( len(images) , num_labels)
            # count batch iterations
            run_batch_count = run_batch_count + 1
            summaries_train, batch_loss, learning_rate, _ = sess.run(
                [summaries.train_merged, lrcn.loss, lrcn.current_lr, lrcn.optimizer],feed_dict=fdict)

            settings.logger.info("Learning rate %2.8f, batch loss : %2.5f " % (learning_rate, batch_loss))

            tboard_writer.add_summary(summaries_train, global_step=dataset.get_global_step())
            tboard_writer.flush()

            if settings.do_validation:
                test_ran = test(dataset, lrcn, settings, sess, tboard_writer, summaries)
                if test_ran:
                    dataset.set_or_swap_phase(defs.phase.train)
        # if an epoch was completed (and not just loaded, do saving and logging)
        if run_batch_count > 0:
            dataset.logger.info("Epoch [%d] training run complete." % (1+epochIdx))
            # save a checkpoint every epoch
            settings.save(sess, dataset, progress="ep_%d_btch_%d" % (1+epochIdx, dataset.batch_index),
                              global_step=dataset.get_global_step())
            dataset.epoch_index = dataset.epoch_index + 1
        # reset phase
        dataset.reset_phase(defs.phase.train)



# test the network on validation data
def test(dataset, lrcn, settings, sess, tboard_writer, summaries):
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


    # validation
    while dataset.loop():
        # get images and labels
        images, ground_truth = dataset.read_next_batch()
        fdict, num_labels = get_feed_dict(lrcn, settings, images, ground_truth)
        dataset.print_iter_info(len(images), num_labels)
        logits = sess.run(lrcn.logits, feed_dict=fdict)
        lrcn.process_validation_logits(logits, dataset, fdict[lrcn.inputLabels])
    # done, get accuracy
    if settings.workflow == defs.workflows.imgdesc:
        # get description metric
        # do an ifthenelse on the evaluation type (eg coco)

        # default eval. should be sth like a json production
        if settings.eval_type == defs.eval_type.coco:
            # evaluate coco
            # format expected is as per  http://mscoco.org/dataset/#format
            # [{ "image_id" : int, "caption" : str, }]

            # get captions from logits, write them in the needed format,
            # pass them to the evaluation function
            ids_captions = dataset.logits_to_captions(lrcn.item_logits)
            if not len(ids_captions[0]) == len(ids_captions[1]):
                settings.logger.error("Unequal number of image ids and captions")
                error("Image ids / captions mismatch")

            json_data = [ { "image_id" : ids_captions[0][i] , "caption" : ids_captions[1][i] }
                          for i in range(len(ids_captions[0]))]
            # write results

            results_file = dataset.input_source_files[defs.val_idx] + ".coco.json"
            dataset.logger.info("Writing captioning results to %s" % results_file)
            with open(results_file , "w") as fp:
                json.dump(json_data, fp)

            # also, get captions from the read image paths - labels files
            # initialize with it the COCO object
            # ....
            settings.logger.info("Evaluating captioning using ground truth file %s" % str(settings.caption_ground_truth))
            command = '$(which python2) tools/python2_coco_eval/coco_eval.py %s %s' % (results_file, settings.caption_ground_truth)
            dataset.logger.debug("evaluation command is [%s]" % command)
            os.system(command)


    else:
        accuracy = lrcn.get_accuracy()

        summaries.val.append(tf.summary.scalar('accuracyVal', accuracy))
        dataset.logger.info("Validation run complete, accuracy: %2.5f" % accuracy)
        tboard_writer.add_summary(summaries.val_merged, global_step=dataset.get_global_step())
    tboard_writer.flush()
    return True



# the main function
def main():

    # create and initialize settings and dataset objects
    settings = Settings()
    settings.initialize(sys.argv)

    # init summaries for printage
    summaries=Summaries()

    # initialize & pre-process dataset
    dataset = dataset_.Dataset()
    dataset.initialize(settings)

    # create and configure the nets : CNN and / or lstm
    lrcn = lrcn_. LRCN()
    lrcn.create(settings, dataset, summaries)

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
        test(dataset, lrcn, settings,  sess, tboard_writer, summaries)

    # mop up
    tboard_writer.close()
    sess.close()
    settings.logger.info("Run [%s] complete." % settings.run_id)

if __name__ == "__main__":
    main()
