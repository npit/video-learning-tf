# generic IO
import pickle
from shutil import copyfile
import sys
import tensorflow as tf
import argparse
from parse_opts import *

# project modules
import lrcn_
import dataset_

# utils
from utils_ import *
from tools.inspect_checkpoint import get_checkpoint_tensor_names
import logging, configparser, json, yaml
from defs_ import defs


# summaries for training & validation
class Summaries:
    train = []
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
    # run mode and type
    run_id = ""
    workflow = defs.workflows.acrec.singleframe

    # save / load configuration
    resume_file = None
    data_path = None
    run_folder = None
    path_prepend_folder = None

    class network:
        # architecture settings
        frame_encoding_layer = "fc7"
        lstm_num_hidden = 256
        num_classes = None


    # data input format
    raw_image_shape = (240, 320, 3)
    mean_image = None
    image_shape = (227, 227, 3)
    data_format = defs.data_format.tfrecord
    frame_format = "jpg"

    class train:
        # training settings
        batch_size_train = 100
        do_training = False
        epochs = 15
        optimizer = defs.optim.sgd
        base_lr = 0.001
        lr_mult = 2
        lr_decay = (defs.decay.granularity.exp, defs.decay.scheme.interval, 1000, 0.96)
        dropout_keep_prob = 0.5
        batch_item = defs.batch_item.default


    class val:
        # validation settings
        do_validation = False
        validation_interval = 1
        batch_size_val = 88
        batch_item = defs.batch_item.default

    class captioning:
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

    def get_num_batches(self):
        if not self.datasets:
            return -1
        return len(self.datasets[0].batches)

    # should resume, if set and non-empty
    def should_resume(self):
        return self.resume_file is not None and self.resume_file

    def read_config(self, config):

        # read global stuff
        self.workflow = defs.check(config['workflow'], defs.workflows)
        self.resume_file = config['resume_file']
        self.run_folder = config["run_folder"]
        self.run_folder = config["run_folder"]

        # read phase information
        self.phase = config["phase"]
        if type(self.phase) != list:
            self.phase = [self.phase]

        # read network  architecture stuff
        self.network = Settings.network()
        self.network.frame_encoding_layer = config["network"]["frame_encoding_layer"]
        self.network.lstm_num_hidden = config["network"]["lstm_num_hidden"]
        self.network.lstm_num_layers = config["network"]["lstm_num_layers"]
        self.network.num_classes = config["network"]["num_classes"]

        frame_fusion = parse_seq(config["network"]["frame_fusion"])
        self.network.frame_fusion = []
        self.network.frame_fusion_type, self.network.frame_fusion_method = \
            defs.check(frame_fusion[0], defs.fusion_type), defs.check(frame_fusion[1], defs.fusion_method)

        clip_fusion = parse_seq(config["network"]["clip_fusion"])
        self.network.clip_fusion = []
        self.network.clip_fusion_type, self.network.clip_fusion_method = \
            defs.check(clip_fusion[0], defs.fusion_type), defs.check(clip_fusion[1], defs.fusion_method)

        self.train, self.val = None, None
        for phase in self.phase:
            phase = defs.check(phase, defs.phase)
            if phase == defs.phase.train:
                # read training opts
                self.train = Settings.train()
                obj = config[phase]
                self.train.batch_size = obj['batch_size']
                self.train.epochs = obj['epochs']
                self.train.base_lr = obj['base_lr']
                self.train.lr_mult = obj['lr_mult']
                self.train.lr_decay = obj['lr_decay']
                self.train.clip_norm = obj['clip_norm']
                self.train.dropout_keep_prob = obj['dropout_keep_prob']
            if phase == defs.phase.val:
                # read validation opts
                self.val = Settings.val()
                obj = config[phase]
                self.val.batch_size = obj['batch_size']
                self.val.logits_save_interval = obj['logits_save_interval']

        # read logging information
        self.save_freq_per_epoch = config['logging']['save_freq_per_epoch']
        self.logging_level = eval(config['logging']['level'])
        if self.logging_level in ['INFO','DEBUG','WARN']:
            error("Invalid logging level: %s:" % (self.logging_level))
        self.tensorboard_folder = config['logging']['tensorboard_folder']
        self.print_tensors = config['logging']['print_tensors']

        # read data sources
        self.datasets = []
        for dataid in config['data']:
            dataobj = config['data'][dataid]

            dset = dataset_.Dataset()
            id = dataid
            path = dataobj['data_path']
            mean_image = parse_seq(dataobj['mean_image'])
            batch_item = defs.check(dataobj['batch_item'], defs.batch_item)
            prepend_folder = dataobj['prepend_folder']
            image_shape = parse_seq(dataobj['image_shape'])
            imgproc_raw = parse_seq(dataobj['imgproc'])
            imgproc = []
            for opt in imgproc_raw:
                imgproc.append(defs.check(opt, defs.imgproc))
            raw_image_shape = parse_seq(dataobj['mean_image'])
            data_format = defs.check(dataobj['data_format'], defs.data_format)
            frame_format = dataobj['frame_format']
            # read image processing params

            do_random_cropping = defs.imgproc.rand_crop in imgproc
            do_center_cropping = defs.imgproc.center_crop in imgproc
            do_resize = defs.imgproc.resize in imgproc
            do_random_mirroring = defs.imgproc.rand_mirror in imgproc

            if raw_image_shape is not None:
                imgproc.append(defs.imgproc.raw_resize)
            if mean_image is not None:
                imgproc.append(defs.imgproc.sub_mean)

            # fix potential inconsistencies
            if sum([do_random_cropping, do_center_cropping, do_resize]) > 1:
                error("Need at most one image processing parameter. Imgproc params : %s" % imgproc)
            if self.val:
                if do_random_cropping or do_random_cropping:
                    if do_random_cropping:
                        warning("Random cropping is enabled in validation mode.")
                    if do_random_mirroring:
                        warning("Random mirroring is enabled in validation mode.")
                    ans = input("continue? (y/n)")
                    if ans != "y":
                        error("Aborted.")

            dset.initialize(id, path, mean_image, prepend_folder, image_shape, imgproc, raw_image_shape, data_format, frame_format, batch_item)
            if self.train:
                dset.calculate_batches(self.train.batch_size)
            elif self.val:
                dset.calculate_batches(self.val.batch_size)

            self.datasets.append(dset)

        # read captioning
        captioning = config['captioning']
        word_embeddings_file = captioning['word_embeddings_file']
        ground_truth = captioning['caption_ground_truth']
        evaluation_type = captioning['eval_type']
        caption_search = captioning['caption_search']
        for dset in self.datasets:
            dset.initialize_workflow(word_embeddings_file)





    # file initialization
    def initialize_from_file(self, init_file):
        if init_file is None:
            return
        if not os.path.exists(init_file):
            error("Unable to read initialization file [%s]." % init_file)
            return

        tag_to_read = "run"
        print("Initializing from file %s" % init_file)
        if init_file.endswith(".ini"):
            config = configparser.ConfigParser()
            config.read(init_file)
            if not config[tag_to_read]:
                error('Expected header [%s] in the configuration file!' % tag_to_read)
            config = config[tag_to_read]
            for var in config:
                exec("self.%s=%s" % (var, config[var]))
        elif init_file.endswith("yml") or init_file.endswith("yaml"):
            with open(init_file,"r") as f:
                config = yaml.load(f)[tag_to_read]
                self.read_config(config)
        # set append the config.ini.xxx suffix to the run id
        trainval_str = ""
        if self.train:
            trainval_str = "train"
        if self.val:
            trainval_str = trainval_str + "val"
        if self.should_resume():
            trainval_str = trainval_str + "_resume"
        else:
            trainval_str = trainval_str + "_scratch"

        # if run id specified, use it
        if self.run_id:
            run_identifiers = [self.workflow ,self.run_id , trainval_str]
        else:
            if init_file == "config.ini":
                run_identifiers = [self.workflow, trainval_str]
            else:
                # use the configuration file suffix
                file_suffix = init_file.split(".")[-1]
                run_identifiers = [self.workflow, file_suffix, trainval_str]

        self.run_id = "_".join(run_identifiers)
        print("Initialized run [%s] from file %s" % (self.run_id, init_file))
        sys.stdout.flush()

    # initialize stuff
    def initialize(self, init_file):

        self.initialize_from_file(init_file)

        if not os.path.exists(self.run_folder):
            warning("Non existent run folder %s - creating." % self.run_folder)
            os.mkdir(self.run_folder)
        # if config file is not in the run folder, copy it there to preserve a settings log
        if not (os.path.basename(init_file) == self.run_folder):
            copyfile(init_file,  os.path.join(self.run_folder, os.path.basename(init_file)))

        # configure the logs
        self.timestamp = get_datetime_str()
        logfile = os.path.join(self.run_folder, "log_" + self.run_id + "_" + self.timestamp + ".log")
        self.logger = CustomLogger()
        CustomLogger.instance = self.logger
        self.logger.configure_logging(logfile, self.logging_level)

        # train-val mode has become unsupported
        if self.train and self.val:
            error("Cannot specify simultaneous training and validation run, for now.")

        # set the tensorboard mode-dependent folder
        mode_folder = defs.phase.train if self.train else defs.phase.val
        self.tensorboard_folder = os.path.join(self.run_folder, self.tensorboard_folder, mode_folder)

        sys.stdout.flush(), sys.stderr.flush()
        # if not resuming, set start folder according to now()
        if  self.should_resume():
            if self.train:
                # load batch and epoch where training left off
                info("Resuming training.")
                # resume training metadata only in training
                self.resume_metadata()
            if self.val:
                info("Evaluating trained network.")
        else:
            if self.train:
                info("Starting training from scratch.")
            else:
                if self.val:
                    warning("Starting validation-only run with an untrained network.")
                else:
                    error("Neither training nor validation is enabled.")

        info("Starting [%s] workflow on folder [%s]." % (self.workflow, self.run_folder))

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

            # inform datasets
            for dset in self.datasets:
                dset.restore(self.train_index, self.epoch_index, self.sequence_length)
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
                    ans = input("Continue? (y/n)")
                    if ans != "y":
                        error("Failed to load checkpoint")
                if names_missing_from_curr:
                    warning("There are checkpoint variables missing in the project:[%s]" % names_missing_from_curr)
                    # Better warn the user and await input
                    ans = input("Continue? (y/n)")
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

# check if we should save
def should_save_now(self, global_step):
    if self.save_interval == None or self.phase != defs.phase.train:
        return False
    return global_step % self.save_interval == 0


# print information on current iteration
def print_iter_info(settings, dataset,  num_images, num_labels, padding):
    padinfo = "(%d padding)" % padding if padding > 0 else ""
    msg = "Mode: [%s], epoch: %2d/%2d, batch %4d / %4d : %3d images%s, %3d labels" % \
          (dataset.batch_index, len(dataset.batches), num_images, padinfo, num_labels)
    if settings.phase == defs.phase.train:
        msg = "Mode: [%s], epoch: %2d/%2d, %s" % (settings.phase, settings.epoch_index + 1, settings.epochs, msg)
    # same as train, but no epoch
    elif settings.phase == defs.phase.val:
        msg = "Mode: [%s], batch %4d / %4d : %3d images%s, %3d labels" % (settings.phase, dataset.batch_index,
                                                                      len(dataset.batches), num_images, padinfo, num_labels)
    info(msg)

# train the network
def train_test(settings, dataset, lrcn, sess, tboard_writer, summaries):
    info("Starting train/test")
    run_batch_count = 0

    for epochIdx in range(dataset.epoch_index, dataset.epochs):
        while dataset.loop():
            # read  batch
            images, ground_truth = dataset.get_next_batch()
            fdict, num_labels, padding = get_feed_dict(lrcn, dataset, images, ground_truth)
            print_iter_info(settings, dataset, len(images), padding, num_labels)

            # count batch iterations
            run_batch_count = run_batch_count + 1
            summaries_train, batch_loss, learning_rate, global_step, _ = sess.run(
                [summaries.train_merged, lrcn.loss, lrcn.current_lr, lrcn.global_step, lrcn.optimizer],feed_dict=fdict)
            # calcluate the number of bits
            nats = batch_loss / math.log(dataset.num_classes)
            info("Learning rate %2.8f, global step: %d, batch loss/nats : %2.5f / %2.3f " % (learning_rate, global_step, batch_loss, nats))
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
    tic = time.time()
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
        info("Validation run complete in [%s], accuracy: %2.5f" % (elapsed_str(tic), accuracy))
        tboard_writer.add_summary(summaries.val_merged, global_step=dataset.get_global_batch_step())
    tboard_writer.flush()
    return True


# the main function
def main(init_file):
    # create and initialize settings and dataset objects
    settings = Settings()
    settings.initialize(init_file)


    # init summaries for printage
    summaries = Summaries()

    # create and configure the nets : CNN and / or lstm
    lrcn = lrcn_.LRCN()
    lrcn.create(settings, summaries)

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
        train_test(settings, lrcn,  sess, tboard_writer, summaries)
    elif settings.do_validation:
        test(lrcn, settings,  sess, tboard_writer, summaries)

    # mop up
    tboard_writer.close()
    sess.close()
    info("Run [%s] complete." % settings.run_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("init_file", help="Configuration .ini file for the run.")
    args = parser.parse_args()

    main(args.init_file)
