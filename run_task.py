
# generic IO
import pickle
from shutil import copyfile
import sys
import tensorflow as tf

# project modules
import lrcn_
import dataset_

# utils
from utils_ import *
from tools.inspect_checkpoint import get_checkpoint_tensor_names
import logging, configparser, json
from defs_ import *


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
    workflow = defs.workflows.acrec.singleframe

    # save / load configuration
    resume_file = None
    data_path = None
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
    lr_decay = (defs.decay.granularity.exp, defs.decay.scheme.interval, 1000, 0.96)
    dropout_keep_prob = 0.5

    # validation settings
    do_validation = False
    validation_interval = 1
    batch_size_val = 88
    batch_item = defs.batch_item.default

    # caption generation settings
    caption_search = defs.caption_search.max
    eval_type = defs.eval_type.coco

    # logging
    logging_level = logging.DEBUG
    tensorboard_folder = "tensorboard_graphs"

    sequence_length = None

    # end of user - settable parameters
    ################################

    # internal variables
    ###############################

    # initialization

    # input data files
    input_files = [[], []]

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

    # set the input files
    def set_input_files(self):
        # for directory run path, use default file names: data.<train|test>                                                                                                              
        if os.path.isdir(self.data_path):
            basefilename = "data"
            self.input_files[defs.train_idx] = os.path.join(self.data_path, basefilename + ".train")
            self.input_files[defs.val_idx] = os.path.join(self.data_path, basefilename + ".test")
            # set the run path to the point to the file
            self.data_path = self.input_files[defs.train_idx] if self.do_training else self.input_files[defs.val_idx]
        else:
            # run path is already the file path
            if self.do_training:
                self.input_files[defs.train_idx] = self.data_path
            else:
                self.input_files[defs.val_idx] =  self.data_path

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
        trainval_str = ""
        if self.do_training:
            trainval_str = "train"
        if self.do_validation:
            trainval_str = trainval_str + "val"
        if self.should_resume():
            trainval_str = trainval_str + "_resume"
        else:
            trainval_str = trainval_str + "_scratch"

        # if run id specified, use it
        if self.run_id:
            run_identifiers = [self.workflow ,self.run_id , trainval_str]
        else:
            if self.init_file == "config.ini":
                run_identifiers = [self.workflow, trainval_str]
            else:
                # use the configuration file suffix
                file_suffix = self.init_file.split(".")[-1]
                run_identifiers = [self.workflow, file_suffix, trainval_str]

        self.run_id = "_".join(run_identifiers)

        # fix potential inconsistencies
        if self.do_validation and self.do_random_cropping:
            warning("Disabling enabled random cropping in validation.")
        if self.do_validation and self.do_random_mirroring:
            warning("Disabling enabled random mirroring in validation.")

        print("Initialized run [%s] from file %s" % ( self.run_id, self.init_file))
        sys.stdout.flush()

    # initialize stuff
    def initialize(self, args):
        if len(args) > 1:
            self.init_file = args[-1]

        self.initialize_from_file()
        if not os.path.exists(self.data_path):
            error("Non existent data file/folder %s" % self.data_path)
        if not os.path.exists(self.run_folder):
            warning("Non existent run folder %s - creating." % self.run_folder)
            os.mkdir(self.run_folder)
        # if config file is not in the run folder, copy it there to preserve a settings log
        if not (os.path.basename(self.init_file) == self.run_folder):
            copyfile(self.init_file,  os.path.join(self.run_folder, os.path.basename( self.init_file)))

        # configure the logs
        self.timestamp = get_datetime_str()
        logfile = os.path.join(self.run_folder, "log_" + self.run_id + "_" + self.timestamp + ".log")
        self.logger = CustomLogger()
        CustomLogger.instance = self.logger
        self.logger.configure_logging(logfile, self.logging_level)

        sys.stdout.flush(), sys.stderr.flush()
        # if not resuming, set start folder according to now()
        if  self.should_resume():
            if self.do_training:
                # load batch and epoch where training left off
                info("Resuming training.")
                # resume training metadata only in training
                self.resume_metadata()
            if self.do_validation:
                info("Evaluating trained network.")
        else:
            if self.do_training:
                info("Starting training from scratch.")
            else:
                if self.do_validation:
                    warning("Starting validation-only run with an untrained network.")
                else:
                    error("Neither training nor validation is enabled.")
        info("Starting [%s] workflow on folder [%s]." % (self.workflow, self.run_folder))
        self.set_input_files()


    # restore dataset meta parameters
    def resume_metadata(self):
        if self.should_resume():
            savefile_metapars = os.path.join(self.run_folder,"checkpoints", self.resume_file + ".snap")
            if not os.path.exists(savefile_metapars):
                error("Graph savefile does not exist: %s" %  savefile_metapars)
            info("Resuming metadata from file:" + savefile_metapars)

            try:
                # load saved parameters pickle
                with open(savefile_metapars, 'rb') as f:
                    params = pickle.load(f)
            except Exception as ex:
                error(ex)

            # set run options from loaded stuff
            self.train_index, self.epoch_index = params[:2]
            if defs.workflows.is_description(self.workflow):
                self.sequence_length = params[2:]

            info("Restored training snapshot of epoch %d, train index %d" % (self.epoch_index+1, self.train_index))

    # restore graph variables
    def resume_graph(self, sess, ignorable_variable_names):
        if self.should_resume():
            if self.saver is None:
                self.saver = tf.train.Saver()

            savefile_graph = os.path.join(self.run_folder,"checkpoints", self.resume_file)
            info("Resuming tf graph from file:" + savefile_graph)
            if not (os.path.exists(savefile_graph + ".meta") or os.path.exists(savefile_graph + ".index")):
                error("Missing meta or index part from graph savefile: %s" % savefile_graph)

            try:
                # if we are in validation mode, the 'global_step' training variable is discardable
                if self.do_validation:
                    ignorable_variable_names.append(defs.variables.global_step)

                chkpt_names = get_checkpoint_tensor_names(savefile_graph)
                # get all variables the project, omitting the :<num> appendices
                curr_names = [ drop_tensor_name_index(t.name) for t in tf.global_variables()]
                names_missing_from_chkpt = [n for n in curr_names if n not in chkpt_names and n not in ignorable_variable_names]
                names_missing_from_curr = [n for n in chkpt_names if n not in curr_names and n not in ignorable_variable_names]

                if names_missing_from_chkpt:
                    missing_unignorable = [n for n in names_missing_from_chkpt if not n in ignorable_variable_names]
                    warning("Unignorable variables missing from checkpoint:[%s]" % missing_unignorable)
                    # Better warn the user and await input
                    ans = input_files("Continue? (y/n)")
                    if ans != "y":
                        error("Failed to load checkpoint")
                if names_missing_from_curr:
                    warning("There are checkpoint variables missing in the project:[%s]" % names_missing_from_curr)
                    # Better warn the user and await input
                    ans = input_files("Continue? (y/n)")
                    if ans != "y":
                        error("Failed to load checkpoint")
                # load saved graph file
                tf.reset_default_graph()
                self.saver.restore(sess, savefile_graph)
            except tf.errors.NotFoundError as err:
                # warning(err.message)
                pass
            except:
                error("Failed to load checkpoint!")

    # save graph and dataset stuff
    def save(self, sess, dataset, progress, global_step):
        try:
            if self.saver is None:
                self.saver = tf.train.Saver()
            # save the graph
            checkpoints_folder = os.path.join(self.run_folder, "checkpoints")
            if not os.path.exists(checkpoints_folder):
                os.makedirs(checkpoints_folder)

            basename = os.path.join(checkpoints_folder, get_datetime_str() + "_" + self.workflow + "_" + progress)
            savefile_graph = basename + ".graph"

            info("Saving graph  to [%s]" % savefile_graph)
            saved_instance_name = self.saver.save(sess, savefile_graph, global_step=global_step)

            # save dataset metaparams
            savefile_metapars = saved_instance_name + ".snap"

            info("Saving params to [%s]" % savefile_metapars)
            info("Saving params for epoch index %d, train index %d" %
                (dataset.epoch_index, dataset.batch_index_train))

            params2save = [dataset.batch_index_train, dataset.epoch_index]
            if defs.workflows.is_description(self.workflow):
                params2save += [dataset.max_caption_length]

            with open(savefile_metapars,'wb') as f:
                pickle.dump(params2save,f)
        except Exception as ex:
            error(ex)


def get_feed_dict(lrcn, dataset, images, ground_truth):
    fdict = {}
    padding = 0
    # supply the images
    fdict[lrcn.inputData] = images

    # for description workflows, supply wordvectors and caption lengths
    if defs.workflows.is_description(dataset.workflow):
        # get words per caption, onehot labels, embeddings
        fdict[lrcn.inputLabels] = ground_truth["onehot_labels"]
        fdict[lrcn.caption_lengths] = ground_truth['caption_lengths']
        fdict[lrcn.word_embeddings] = ground_truth['word_embeddings']
        fdict[lrcn.non_padding_word_idxs] = ground_truth['non_padding_index']
        num_labels = len(ground_truth["onehot_labels"])
    else:
        fdict[lrcn.inputLabels] = ground_truth
        num_labels = len(ground_truth)

    return fdict, num_labels, padding


# train the network
def train_test(settings, dataset, lrcn, sess, tboard_writer, summaries):
    info("Starting train/test")
    run_batch_count = 0

    for epochIdx in range(dataset.epoch_index, dataset.epochs):
        dataset.set_or_swap_phase(defs.phase.train)
        while dataset.loop():
            # read  batch
            images, ground_truth = dataset.get_next_batch()
            fdict, num_labels, padding = get_feed_dict(lrcn, dataset, images, ground_truth)
            dataset.print_iter_info( len(images), padding, num_labels)
            # count batch iterations
            run_batch_count = run_batch_count + 1
            summaries_train, batch_loss, learning_rate, global_step, _ = sess.run(
                [summaries.train_merged, lrcn.loss, lrcn.current_lr, lrcn.global_step, lrcn.optimizer],feed_dict=fdict)

            info("Learning rate %2.8f, global step: %d, batch loss : %2.5f " % (learning_rate, global_step, batch_loss))
            info("Dataset global step %d, epoch index %d, batch size train %d, batch index train %d" %
                                 (dataset.get_global_batch_step(), dataset.epoch_index, dataset.batch_size_train, dataset.batch_index_train))

            tboard_writer.add_summary(summaries_train, global_step=dataset.get_global_batch_step())
            tboard_writer.flush()

            if settings.do_validation:
                test_ran = test(dataset, lrcn, settings, sess, tboard_writer, summaries)
                if test_ran:
                    dataset.set_or_swap_phase(defs.phase.train)

            # check if we need to save
            if dataset.should_save_now(global_step):
                # save a checkpoint if needed
                settings.save(sess, dataset, progress="ep_%d_btch_%d_gs_%d" % (1 + epochIdx, dataset.batch_index, global_step),
                              global_step=dataset.get_global_batch_step())
        # if an epoch was completed (and not just loaded, do saving and logging)
        if run_batch_count > 0:
            info("Epoch [%d] training run complete." % (1+epochIdx))
        else:
            info("Resumed epoch [%d] is already complete." % (1+epochIdx))

        dataset.epoch_index = dataset.epoch_index + 1
        # reset phase
        dataset.reset_phase(defs.phase.train)

    # if we did not save already, do it now at the end of training
    if run_batch_count and not dataset.should_save_now(global_step):
        info("Saving model checkpoint out of turn, since training's finished.")
        settings.save(sess, dataset, progress="ep_%d_btch_%d_gs_%d" % (1 + epochIdx, len(dataset.batches), global_step),
                      global_step=dataset.get_global_batch_step())



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
        images, ground_truth = dataset.get_next_batch()
        fdict, num_labels, padding = get_feed_dict(lrcn, dataset, images, ground_truth)
        dataset.print_iter_info(len(images), padding, num_labels)
        logits = sess.run(lrcn.logits, feed_dict=fdict)
        lrcn.process_validation_logits(logits, dataset, fdict, padding)
        lrcn.save_validation_logits_chunk()
    # save the complete output logits
    lrcn.save_validation_logits_chunk(save_all = True)

    # done, get accuracy
    if defs.workflows.is_description(settings.workflow):
        # get description metric
        # do an ifthenelse on the evaluation type (eg coco)

        # default eval. should be sth like a json production
        if settings.eval_type == defs.eval_type.coco:
            # evaluate coco
            # format expected is as per  http://mscoco.org/dataset/#format
            # [{ "image_id" : int, "caption" : str, }]

            # get captions from logits, write them in the needed format,
            # pass them to the evaluation function
            ids_captions = []
            num_processed_logits = 0

            for idx in range(lrcn.validation_logits_save_counter):
                logits_chunk = lrcn.load_validation_logits_chunk(idx)
                ids_captions_chunk = dataset.validation_logits_to_captions(logits_chunk, num_processed_logits)
                ids_captions.extend(ids_captions_chunk)
                num_processed_logits  += len(logits_chunk)
                info("Processed saved chunk %d/%d containing %d items - item total: %d" %
                    (idx+1,lrcn.validation_logits_save_counter, len(logits_chunk), num_processed_logits))
            if len(lrcn.item_logits) > 0:
                error("Should never get item logits last chunk in runtask!!")
                ids_captions_chunk = dataset.validation_logits_to_captions(lrcn.item_logits, num_processed_logits)
                ids_captions.extend(ids_captions_chunk)
                info("Processed existing chunk containing %d items - item total: %d" % (len(lrcn.item_logits), len(ids_captions)))

            # check for erroneous duplicates
            dupl = [obj["image_id"] for obj in ids_captions]
            if duplicates(dupl):
                error("Duplicate image ids in coco validation: %s" % str(dupl))

            # write results
            results_file = dataset.input_source_files[defs.val_idx] + ".coco.results.json"
            info("Writing captioning results to %s" % results_file)
            with open(results_file , "w") as fp:
                json.dump(ids_captions, fp)

            # also, get captions from the read image paths - labels files
            # initialize with it the COCO object
            # ....
            info("Evaluating captioning using ground truth file %s" % str(settings.caption_ground_truth))
            command = '$(which python2) tools/python2_coco_eval/coco_eval.py %s %s' % (results_file, settings.caption_ground_truth)
            debug("evaluation command is [%s]" % command)
            os.system(command)


    else:
        accuracy = lrcn.get_accuracy()
        summaries.val.append(tf.summary.scalar('accuracyVal', accuracy))
        info("Validation run complete, accuracy: %2.5f" % accuracy)
        tboard_writer.add_summary(summaries.val_merged, global_step=dataset.get_global_batch_step())
    tboard_writer.flush()
    return True

# the main function
def main():

    # create and initialize settings and dataset objects
    settings = Settings()
    settings.initialize(sys.argv)

    # init summaries for printage
    summaries = Summaries()

    # initialize & pre-process dataset
    dataset = dataset_.Dataset()
    dataset.initialize(settings)

    # create and configure the nets : CNN and / or lstm
    lrcn = lrcn_.LRCN()
    lrcn.create(settings, dataset, summaries)

    # view_print_tensors(lrcn,dataset,settings,lrcn.print_tensors)

    # create and init. session and visualization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # restore graph variables,
    settings.resume_graph(sess, lrcn.get_ignorable_variable_names())

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
    info("Run [%s] complete." % settings.run_id)

if __name__ == "__main__":
    main()
