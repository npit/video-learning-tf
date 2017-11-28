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
    dataset_per_phase = {}
    global_step = 0

    class network:
        # architecture settings
        frame_encoding_layer = "fc7"
        lstm_num_hidden = 256
        num_classes = None

    class train:
        # training settings
        batch_size_train = 100
        epochs = 15
        epoch_index = 0
        optimizer = defs.optim.sgd
        base_lr = 0.001
        lr_mult = 2
        lr_decay = (defs.decay.granularity.exp, defs.decay.scheme.interval, 1000, 0.96)
        dropout_keep_prob = 0.5
        batch_item = defs.batch_item.default

    class val:
        # validation settings
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

    # logging
    logger = None

    # misc
    saver = None


    def get_train_str(self):
        tr = self.train
        infostr = "epochs: %d, optim: %s" % (tr.epochs, tr.optimizer)

        lrstr = ", lr: [%2.2f," % tr.base_lr
        if tr.lr_mult is not None: lrstr += " mult: %2.2f," % tr.lr_mult
        if tr.lr_decay is not None: lrstr += " %s]" % ", ".join([str(x) for x in tr.lr_decay])
        else: lrstr += " static]"

        infostr += lrstr

        if tr.dropout_keep_prob is not None: infostr += " dropout: %2.2f," % tr.dropout_keep_prob
        if tr.clip_norm is not None:
            infostr += " clip: %2.2f" % tr.clip_norm

        return infostr

    def validation_logits_to_captions(self, logits_chunk, num_processed_logits):
        return self.datasets[self.phase].validation_logits_to_captions(logits_chunk, num_processed_logits)

    def get_batch_sizes(self):
        batch_sizes = []
        for dset in self.datasets[self.phase]:
            batch_sizes.append(dset.batch_size)
        return  batch_sizes

    def compute_save_interval(self):
        # just check the first
        for dset in self.datasets[self.phase]:
            return dset.compute_dataset_portion(self.save_freq_per_epoch)

    def get_batch_index(self):
        return self.datasets[self.phase][0].batch_index

    def rewind_datasets(self):
        for dset in self.datasets[self.phase]:
            dset.rewind()
    def get_datasets(self):
        return self.datasets[self.phase]

    def get_next_batch(self):
        images, ground_truth, ids = [],[],[]
        for dset in self.datasets[self.phase]:
            im, g = dset.get_next_batch()
            images.append(im)
            ground_truth.append(g)
            ids.append(dset.id)
        return images, ground_truth, ids

    def loop(self):
        # it is assumed that all datasets are loop-synchronized
        return self.datasets[self.phase][0].loop()

    def get_dataset_by_tag(self, tag):
        dsets = [dset for dset in self.datasets[self.phase] if dset.tag == tag]
        return dsets

    def get_num_batches(self):
        if not self.datasets:
            return -1
        return len(self.datasets[self.phase][0].batches)

    # should resume, if set and non-empty
    def should_resume(self):
        return self.resume_file is not None and self.resume_file

    def read_config(self, config):

        # read global stuff
        self.workflow = defs.check(config['workflow'], defs.workflows)
        self.resume_file = config['resume_file']
        self.run_folder = config["run_folder"]

        # read phase information
        self.phases = defs.check(config["phase"], defs.phase)
        if type(self.phases) != list:
            self.phases = [self.phases]
        self.phase = self.phases[0]

        # read network  architecture stuff
        self.network = Settings.network()
        if self.workflow == defs.workflows.acrec.multi:
            self.network.multi_workflow = defs.check(config['network']['multi_workflow'], defs.workflows.multi)
        self.network.load_weights = config['network']['load_weights']
        self.network.image_shape = parse_seq(config["network"]["image_shape"])
        self.network.frame_encoding_layer = config["network"]["frame_encoding_layer"]
        self.network.lstm_params = parse_seq(config['network']['lstm_params'])
        if len(self.network.lstm_params) != 3:
            error("Expected lstm params are [num_hidden, num_layers, fusion_method]")
        self.network.lstm_params[2] = defs.check(self.network.lstm_params[2], defs.fusion_method)
        self.network.num_classes = config["network"]["num_classes"]
        dataset_fusion = parse_seq(config["network"]["dataset_fusion"])
        self.network.dataset_fusion_type, self.network.dataset_fusion_method = \
            defs.check(dataset_fusion[0], defs.fusion_type), defs.check(dataset_fusion[1], defs.fusion_method),

        frame_fusion = parse_seq(config["network"]["frame_fusion"])
        self.network.frame_fusion_type, self.network.frame_fusion_method = \
            defs.check(frame_fusion[0], defs.fusion_type), defs.check(frame_fusion[1], defs.fusion_method)

        clip_fusion = parse_seq(config["network"]["clip_fusion"])
        self.network.clip_fusion_type, self.network.clip_fusion_method = \
            defs.check(clip_fusion[0], defs.fusion_type), defs.check(clip_fusion[1], defs.fusion_method)

        self.train, self.val = None, None
        for phase in self.phases:
            if phase == defs.phase.train:
                # read training opts
                self.train = Settings.train()
                obj = config[phase]
                self.train.batch_size = int(obj['batch_size'])
                self.train.epochs = int(obj['epochs'])
                self.train.optimizer = defs.check(obj['optimizer'], defs.optim)
                self.train.base_lr = float(obj['base_lr'])
                self.train.lr_mult = float(obj['lr_mult']) if obj['lr_mult'] != 'None' else None
                lr_decay = parse_seq(obj['lr_decay'])
                self.train.lr_decay = []
                self.train.lr_decay.append(defs.check(lr_decay[0],defs.decay.granularity))
                self.train.lr_decay.append(defs.check(lr_decay[1],defs.decay.scheme))
                self.train.lr_decay.append(int(lr_decay[2]))
                self.train.lr_decay.append(float(lr_decay[3]))
                self.train.clip_norm = int(obj['clip_norm'])
                self.train.dropout_keep_prob = float(obj['dropout_keep_prob'])
            if phase == defs.phase.val:
                # read validation opts
                self.val = Settings.val()
                obj = config[phase]
                self.val.batch_size = int(obj['batch_size'])
                self.val.logits_save_interval = int(obj['logits_save_interval'])

        # read logging information
        self.save_freq_per_epoch = config['logging']['save_freq_per_epoch']
        self.logging_level = config['logging']['level']
        loglevels = ['logging.' + x for x in ['INFO','DEBUG','WARN']]
        if not self.logging_level in loglevels:
            error("Invalid logging level: %s" % (self.logging_level))
        self.logging_level = eval(self.logging_level)
        self.tensorboard_folder = config['logging']['tensorboard_folder']
        self.print_tensors = config['logging']['print_tensors']

        # read data sources
        self.datasets = {}
        for dataid in config['data']:
            dataobj = config['data'][dataid]

            dset = dataset_.Dataset()
            id = dataid
            # phase to run the dataset in
            dataset_phase = defs.check(dataobj['phase'], defs.phase)
            if not dataset_phase in self.phases:
                info("Omitting dataset [%s] due to its phase [%s]" % (id, dataset_phase))
                continue

            if not dataset_phase in self.datasets:
                self.datasets[dataset_phase] = []
            self.datasets[dataset_phase].append(dset)

            # imgproc options
            path = dataobj['data_path']
            mean_image = parse_seq(dataobj['mean_image'])
            batch_item = defs.check(dataobj['batch_item'], defs.batch_item)
            prepend_folder = dataobj['prepend_folder']
            image_shape = parse_seq(dataobj['image_shape'])
            imgproc_raw = parse_seq(dataobj['imgproc'])
            imgproc = []
            for opt in imgproc_raw:
                imgproc.append(defs.check(opt, defs.imgproc))
            if defs.imgproc.raw_resize in imgproc and not mean_image:
                error("[%s] option requires a supplied mean image intensity." % defs.imgproc.raw_resize)
            raw_image_shape = parse_seq(dataobj['raw_image_shape'])
            data_format = defs.check(dataobj['data_format'], defs.data_format)
            frame_format = dataobj['frame_format']
            tag = defs.check(dataobj['tag'], defs.dataset_tag)

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
                    ans = input("continue? (y/n) : ")
                    if ans != "y":
                        error("Aborted.")
            dset.initialize(id, path, mean_image, prepend_folder, image_shape, imgproc, raw_image_shape, data_format,
                                frame_format, batch_item, self.network.num_classes, tag)

            # read captioning - it's dataset-dependent
            if 'captioning' in dataobj:
                captioning = dataobj['captioning']
                word_embeddings_file = captioning['word_embeddings_file']
                ground_truth = captioning['caption_ground_truth']
                evaluation_type = captioning['eval_type']
                caption_search = captioning['caption_search']
                dset.initialize_workflow(word_embeddings_file)

    def initialize_datasets(self):
        if not self.datasets:
            error("No dataset configured to active phase [%s]" % self.phase)
        self.input_mode = defs.input_mode.get_from_workflow(self.workflow)
        for phase in self.phases:
            for dset in self.datasets[phase]:
                if self.train:
                    dset.calculate_batches(self.train.batch_size, self.input_mode)
                elif self.val:
                    dset.calculate_batches(self.val.batch_size, self.input_mode)
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
        info("Initialized from configuration file: [%s]" % init_file)

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

        # initialize datasets
        self.initialize_datasets()

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

            if self.resume_file == defs.names.latest_savefile:
                with open(os.path.join(self.run_folder,"checkpoints","checkpoint"),"r") as f:
                    for line in f:
                        savefile_graph = line.strip().split()[-1].strip()
                        if savefile_graph[::len(savefile_graph)-1] == '""': savefile_graph = savefile_graph[1:-1]
                        savefile_metapars = savefile_graph + ".snap"
                        msg = "Resuming latest tf metadata: [%s]" % savefile_metapars
                        break
            else:
                savefile_metapars = self.resume_file + ".snap"
                msg = "Resuming specified tf metadata: [%s]" % savefile_metapars

            info(msg)
            if not os.path.exists(savefile_metapars):
                error("Metaparameters savefile does not exist: %s" %  savefile_metapars)
            try:
                # load saved parameters pickle
                with open(savefile_metapars, 'rb') as f:
                    params = pickle.load(f)
            except Exception as ex:
                error(ex)

            # set run options from loaded stuff
            batch_info, self.train.epoch_index = params[:2]
            # assign global step
            try:
                self.global_step = params[2]
            except:
                # parse from filename
                global_step_str = os.path.basename(savefile_metapars).split(".")[-2].split("-")[-1]
                self.global_step = int(global_step_str)

            if defs.workflows.is_description(self.workflow):
                self.sequence_length = params[2:]

            # inform datasets - if batch index info is paired with a dataset id, inform that dataset. Else, inform the 1st
            for dset in self.get_datasets():
                idx = 0
                if type(batch_info) == dict:
                    if dset.tag in batch_info:
                        idx = batch_info[dset.tag]
                else:
                    # an int - update it regardless
                    idx = batch_info
                dset.restore(idx, self.train.epoch_index, self.sequence_length)

            info("Restored training snapshot of epoch %d, train index %s, global step %d" % (self.train.epoch_index+1, str(batch_info), self.global_step))

    # restore graph variables
    def resume_graph(self, sess, ignorable_variable_names):
        if self.should_resume():
            if self.saver is None:
                self.saver = tf.train.Saver()

            if self.resume_file == defs.names.latest_savefile:
                with open(os.path.join(self.run_folder,"checkpoints","checkpoint"),"r") as f:
                    for line in f:
                        savefile_graph = line.strip().split()[-1].strip()
                        if savefile_graph[::len(savefile_graph)-1] == '""': savefile_graph = savefile_graph[1:-1]
                        msg = "Resuming latest tf graph: [%s]" % savefile_graph
                        break
            else:
                savefile_graph = os.path.join(self.resume_file)
                msg = "Resuming specified tf graph: [%s]" % savefile_graph

            info(msg)
            if not (os.path.exists(savefile_graph + ".meta")
                    and os.path.exists(savefile_graph + ".index")
                    and os.path.exists(savefile_graph + ".snap")):
                error("Missing meta, snap or index part from graph savefile: %s" % savefile_graph)

            try:
                # if we are in validation mode, the 'global_step' training variable is discardable
                if self.val:
                    ignorable_variable_names.append(defs.names.global_step)

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
    def save(self, sess, progress):
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
            saved_instance_name = self.saver.save(sess, savefile_graph, global_step=self.global_step)

            # save dataset metaparams
            savefile_metapars = saved_instance_name + ".snap"

            info("Saving params to [%s]" % savefile_metapars)
            info("Saving params for epoch index %d, train index %d" %
                (self.train.epoch_index + 1, self.get_batch_index()))

            params2save = [self.get_batch_index(), self.train.epoch_index, self.global_step]
            if defs.workflows.is_description(self.workflow):
                params2save += [ dat.max_caption_length for dat in self.get_datasets()]

            with open(savefile_metapars,'wb') as f:
                pickle.dump(params2save,f)
        except Exception as ex:
            error(ex)

def get_feed_dict(lrcn, settings, images, ground_truth, dataset_ids):
    fdict = {}
    for required_input in lrcn.input:
        i_tens, i_type, i_datatag = required_input
        dataset_id = [ dset.id for dset in settings.get_datasets() if dset.tag == i_datatag]
        if not (len(dataset_id) == 1):
            error("%d datasets satisfy the following network input requirement, but exactly one must. %s." % (len(dataset_id), str(required_input)))
        dataset_idx = dataset_ids.index(dataset_id[0])
        if i_type == defs.net_input.visual:
            fdict[i_tens] = images[dataset_idx]
        elif i_type == defs.net_input.labels:
            fdict[i_tens] = ground_truth[dataset_idx]
            num_labels = len(ground_truth[dataset_idx])

    padding = 0
    # for description workflows, supply wordvectors and caption lengths
    if defs.workflows.is_description(settings.workflow):
        # get words per caption, onehot labels, embeddings
        fdict[lrcn.inputLabels] = ground_truth["onehot_labels"]
        fdict[lrcn.caption_lengths] = ground_truth['caption_lengths']
        fdict[lrcn.word_embeddings] = ground_truth['word_embeddings']
        fdict[lrcn.non_padding_word_idxs] = ground_truth['non_padding_index']
        num_labels = len(ground_truth["onehot_labels"])
    #else:
    #    fdict[lrcn.inputLabels] = ground_truth
    #    num_labels = len(ground_truth)

    return fdict, num_labels, padding

# check if we should save
def should_save_now(self, global_step):
    if self.save_interval == None or self.phase != defs.phase.train:
        return False
    return global_step % self.save_interval == 0

# print information on current iteration
def print_iter_info(settings, num_images, num_labels, padding):
    dataset = settings.datasets[settings.phase][0]
    padinfo = "(%d padding)" % padding if padding > 0 else ""
    epoch_str = "" if settings.val else "epoch: %2d/%2d," % (settings.train.epoch_index+1, settings.train.epochs)
    msg = "Mode: [%s], %s batch %4d / %4d : %s images%s, %3d labels" % \
          (settings.phase, epoch_str , dataset.batch_index, len(dataset.batches), str(num_images), padinfo, num_labels)
    info(msg)

# train the network
def train_test(settings, lrcn, sess, tboard_writer, summaries):
    run_batch_count = 0

    save_interval = settings.compute_save_interval()
    info("Starting train")
    for _ in range(settings.train.epoch_index, settings.train.epochs):
        while settings.loop():
            # read  batch
            images, ground_truth, dataset_ids = settings.get_next_batch()
            fdict, num_labels, padding = get_feed_dict(lrcn, settings, images, ground_truth, dataset_ids)
            print_iter_info(settings, [len(im) for im in images], num_labels, padding)

            # count batch iterations
            run_batch_count += 1
            summaries_train, batch_loss, learning_rate, settings.global_step, _ = sess.run(
                [summaries.train_merged, lrcn.loss, lrcn.current_lr, lrcn.global_step, lrcn.optimizer],feed_dict=fdict)
            # calcluate the number of bits
            nats = batch_loss / math.log(settings.network.num_classes)
            info("Learning rate %2.8f, global step: %d, batch loss/nats : %2.5f / %2.3f " % \
                 (learning_rate, settings.global_step, batch_loss, nats))
            info("Dataset global step %d, epoch index %d, batch sizes %s, batch index train %d" %
                                 (settings.global_step, settings.train.epoch_index + 1, str(settings.get_batch_sizes()),
                                  settings.get_batch_index()))

            tboard_writer.add_summary(summaries_train, global_step=settings.global_step)
            tboard_writer.flush()

            # check if we need to save
            if run_batch_count % save_interval == 0:
                # save a checkpoint if needed
                settings.save(sess, progress="ep_%d_btch_%d_gs_%d" % (1 + settings.train.epoch_index,
                                                                      settings.get_batch_index(), settings.global_step))
        # print finished epoch information
        if run_batch_count > 0:
            info("Epoch [%d] training run complete." % (1+settings.train.epoch_index))
        else:
            info("Resumed epoch [%d] is already complete." % (1+settings.train.epoch_index))

        settings.train.epoch_index = settings.train.epoch_index + 1
        # reset phase
        settings.rewind_datasets()

    # if we did not save already at the just completed batch, do it now at the end of training
    if run_batch_count > 0 and not (run_batch_count %  save_interval == 0):
        info("Saving model checkpoint out of turn, since training's finished.")
        settings.save(sess,  progress="ep_%d_btch_%d_gs_%d" %
                (1 + settings.train.epoch_index, settings.get_num_batches(), settings.global_step))



# test the network on validation data
def test(lrcn, settings, sess, tboard_writer, summaries):
    tic = time.time()

    settings.global_step = 0

    # validation
    while settings.loop():
        # get images and labels
        images, ground_truth, dataset_ids = settings.get_next_batch()
        fdict, num_labels, padding = get_feed_dict(lrcn, settings, images, ground_truth, dataset_ids)
        print_iter_info(settings, [len(im) for im in images], num_labels, padding)
        logits = sess.run(lrcn.logits, feed_dict=fdict)
        lrcn.process_validation_logits( defs.dataset_tag.main, settings, logits, fdict, padding)
        lrcn.save_validation_logits_chunk()
        settings.global_step += 0
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
                ids_captions_chunk = settings.validation_logits_to_captions(logits_chunk, num_processed_logits)
                ids_captions.extend(ids_captions_chunk)
                num_processed_logits  += len(logits_chunk)
                info("Processed saved chunk %d/%d containing %d items - item total: %d" %
                    (idx+1,lrcn.validation_logits_save_counter, len(logits_chunk), num_processed_logits))
            if len(lrcn.item_logits) > 0:
                error("Should never get item logits last chunk in runtask!!")
                ids_captions_chunk = settings.validation_logits_to_captions(lrcn.item_logits, num_processed_logits)
                ids_captions.extend(ids_captions_chunk)
                info("Processed existing chunk containing %d items - item total: %d" % (len(lrcn.item_logits), len(ids_captions)))

            # check for erroneous duplicates
            dupl = [obj["image_id"] for obj in ids_captions]
            if duplicates(dupl):
                error("Duplicate image ids in coco validation: %s" % str(dupl))

            # write results
            results_file = os.path.join(settings.run_folder, "coco.results.json")
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
        tboard_writer.add_summary(summaries.val_merged, global_step= settings.global_step)
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
    if settings.train:
        train_test(settings, lrcn,  sess, tboard_writer, summaries)
    elif settings.val:
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
