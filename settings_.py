from defs_ import *
from utils_ import *
import pickle
from shutil import copyfile
import sys
import tensorflow as tf
from parse_opts import *
import logging, yaml
from feeder import Feeder

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

class Network:
    # architecture settings
    description = "Network representative class"

class Settings:
    # user - settable parameters
    ################################
    # run mode and type
    run_id = ""

    # save / load configuration
    resume_file = None
    save_interval = None
    data_path = None
    run_folder = None
    path_prepend_folder = None
    global_step = 0

    feeder = None

    pipelines = {}
    pipeline_names = []



    class train:
        # training settings
        batch_size = 100
        epochs = 15
        epoch_index = 0
        optimizer = defs.optim.sgd
        base_lr = 0.001
        lr_mult = 2
        lr_decay = (defs.decay.exp, defs.periodicity.interval, 1000, 0.96)
        dropout_keep_prob = 0.5
        batch_item = defs.batch_item.default

    class val:
        # validation settings
        validation_interval = 1
        batch_size = 88
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
    email_notify = None

    # misc
    saver = None

    def get_batch_size(self):
        if defs.phase.train in self.phases:
            return self.train.batch_size
        else:
            return self.val.batch_size

    def get_dropout(self):
        if self.phase == defs.phase.train:
            return self.train.dropout_keep_prob
        return 0.0

    def get_train_str(self):
        tr = self.train
        infostr = "classes: %d, epochs: %d, optim: %s" % (self.num_classes, tr.epochs, tr.optimizer)

        lrstr = ", lr: [%2.2f," % tr.base_lr
        if tr.lr_mult is not None: lrstr += " mult: %2.2f," % tr.lr_mult
        if tr.lr_decay is not None: lrstr += " %s]" % ", ".join([str(x) for x in tr.lr_decay])
        else: lrstr += " static]"

        infostr += lrstr

        if tr.dropout_keep_prob is not None: infostr += " dropout: %2.2f," % tr.dropout_keep_prob
        if tr.clip_norm is not None:
            infostr += " clip: %2.2f" % tr.clip_norm

        return infostr

    def get_val_str(self):
        infostr = "clip_fusion: [%s, %s], logit save: [%d]" % \
                  (self.network.clip_fusion_type, self.network.clip_fusion_method, self.val.logits_save_interval)
        return infostr


    def read_field(self, config, fieldname, validate = None, required = False, listify = False):
        debug("Reading field [%s]" % fieldname)
        # keep track of pipeline fields read
        self.pipeline_field_cache.append(fieldname)
        val = None
        if fieldname in config:
            val = config[fieldname]

        if fieldname not in config or val == None:
            if required:
                error("No default value specified for missing field [%s]" % fieldname)
            else:
                if listify:
                    val = [None]
                return val


        if validate is not None:
            if type(validate) in [list, tuple]:
                if len(validate) != len(val):
                    error("Field [%s] required %d entries, found: [%s]" % (fieldname, len(validate), str(val)))
                for i, (el, v) in enumerate(zip(val, validate)):
                    val[i] = defs.check(el,v)
            else:
                val = defs.check(val, validate)

        if listify:
            if type(val) not in [list, tuple]:
                val = [val]
        return val


    def read_network(self, pipeline_content):
        network = Network()
        self.pipeline_field_cache = []
        # input has to be either a dataset or a pipeline output - if it's a pipeline, if that has not been read, bail and retry later
        network.input = self.read_field(pipeline_content, 'input', validate=None, listify=True)
        if any([x is None for x in network.input]):
            error("<None> or undefined <input> tag in pipeline: %s" % pipeline_content)
        for inp in network.input:
            if inp not in self.pipelines:
                idx = network.input.index(inp)
                is_dataset_tag, tagname = defs.check(inp,  defs.dataset_tag, do_boolean = True)
                if is_dataset_tag:
                    network.input[idx] = tagname
                else:
                    error("Input identifier [%s] is not a dataset tag, but no such pipeline has been declared yet." % (inp))

        network.representation = self.read_field(pipeline_content, 'representation', required = True, validate = defs.representation)
        if network.representation == defs.representation.dcnn:
            network.frame_encoding_layer = self.read_field(pipeline_content, 'frame_encoding_layer', required = True)
        if network.representation == defs.representation.fc:
            network.fc_output_dim = self.read_field(pipeline_content, 'fc_output_dim', required = True)

        network.classifier = self.read_field(pipeline_content, 'classifier', validate = defs.classifier)
        if network.classifier == defs.classifier.lstm:
            params = self.read_field(pipeline_content,"lstm_params")
            network.lstm_params = [int(params[0]), int(params[1]), defs.check(params[2], defs.fusion_method)]

        network.weights_file = self.read_field(pipeline_content, 'weights_file')
        network.frame_fusion = self.read_field(pipeline_content, 'frame_fusion', validate = (defs.fusion_type, defs.fusion_method))
        network.input_shape = self.read_field(pipeline_content, 'input_shape', validate=None, listify=True)
        network.input_fusion = self.read_field(pipeline_content, 'input_fusion', validate=defs.fusion_method)
        for i in range(len(network.input_shape)):
            shp = network.input_shape[i]
            if shp == "None": network.input_shape[i] = None
            elif shp == None: pass
            else: network.input_shape[i] = parse_seq(shp)

        # check for unread pipeline fields
        unread_fields = [x for x in pipeline_content if x not in self.pipeline_field_cache]
        if unread_fields:
            error("Undefined pipeline field(s):" + str(unread_fields))
        return network

    def read_config(self, config, init_file):
        # read global stuff
        self.resume_file = config['resume_file']
        self.run_folder = config["run_folder"]

        if "run_id" in config:
            self.run_id = config["run_id"]

        # read phase information
        self.train, self.val = None, None
        self.phases = defs.check(config["phase"], defs.phase)
        if type(self.phases) != list:
            self.phases = [self.phases]
        self.phase = self.phases[0]

        trainval_str = ""
        if defs.phase.train in self.phases:
            trainval_str = "train"
        if defs.phase.val in self.phases:
            trainval_str = trainval_str + "val"
        if self.should_resume():
            trainval_str = trainval_str + "_resume"
        else:
            trainval_str = trainval_str + "_scratch"
        if self.run_id:
            run_identifiers = [self.run_id , trainval_str]
        else:
            # use the configuration filename
            run_identifiers = [ os.path.basename(init_file), trainval_str]
        self.run_id = "_".join(run_identifiers)

        if not os.path.exists(self.run_folder):
            warning("Non existent run folder %s - creating." % self.run_folder)
            os.mkdir(self.run_folder)

        # read logging information
        self.save_freq_per_epoch = config['logging']['save_freq_per_epoch']
        self.logging_level = config['logging']['level']
        loglevels = ['logging.' + x for x in ['INFO','DEBUG','WARN']]
        if not self.logging_level in loglevels:
            error("Invalid logging level: %s" % (self.logging_level))
        self.tensorboard_folder = config['logging']['tensorboard_folder']
        self.print_tensors = config['logging']['print_tensors']
        self.email_notify = config['logging']['email_notify']
        if self.email_notify:
            self.email_notify = prep_email(self.email_notify)
        self.configure_logging()


        # read network architecture stuff
        pipelines = list(reversed(config['network']['pipelines']))
        while pipelines:
            pipeline = pipelines.pop()
            pname, content = list(pipeline.items())[0]
            debug("Reading network [%s]" % (pname))
            parsed_pipeline = self.read_network(content)
            self.pipelines[pname] = parsed_pipeline
            self.pipeline_names.append(pname)

        self.num_classes = config["network"]["num_classes"]

        for phase in self.phases:
            print(phase, self.phases)
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
                self.train.lr_decay.append(defs.check(lr_decay[0],defs.decay))
                self.train.lr_decay.append(defs.check(lr_decay[1],defs.periodicity))
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
                clip_fusion = parse_seq(obj["clip_fusion"])
                self.val.clip_fusion_type, self.val.clip_fusion_method = \
                    defs.check(clip_fusion[0], defs.fusion_type), defs.check(clip_fusion[1], defs.fusion_method)


        # read data sources - assume video mode by default
        self.feeder = Feeder(defs.input_mode.video, self.phases, (self.train, self.val), self.save_freq_per_epoch, self.run_folder, self.should_resume())

        for dataid in config['data']:
            dataobj = config['data'][dataid]

            id = dataid
            # phase to run the dataset in
            dataset_phase = defs.check(dataobj['phase'], defs.phase)
            if not dataset_phase in self.phases:
                info("Omitting dataset [%s] due to its phase [%s]" % (id, dataset_phase))
                continue

            # imgproc options
            path = dataobj['data_path']
            mean_image = parse_seq(dataobj['mean_image']) if 'mean_image' in dataobj else None
            batch_item = defs.check(dataobj['batch_item'], defs.batch_item) if 'batch_item' in dataobj else defs.batch_item.default
            prepend_folder = dataobj['prepend_folder'] if 'prepend_folder' in dataobj else None
            image_shape = parse_seq(dataobj['image_shape']) if 'image_shape' in dataobj else None
            imgproc_raw = parse_seq(dataobj['imgproc']) if 'imgproc' in dataobj else []
            imgproc = []
            for opt in imgproc_raw:
                imgproc.append(defs.check(opt, defs.imgproc))
            if defs.imgproc.sub_mean in imgproc and not mean_image:
                error("[%s] option requires a supplied mean image intensity." % defs.imgproc.raw_resize)
            raw_image_shape = parse_seq(dataobj['raw_image_shape']) if 'raw_image_shape' in dataobj else None
            data_format = defs.check(dataobj['data_format'], defs.data_format)
            frame_format = dataobj['frame_format'] if 'frame_format' in dataobj else None
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

            val = int(dataobj["read_tries"]) if "read_tries" in dataobj else 1
            read_tries = val

            if 'captioning' in dataobj:
                captioning = dataobj['captioning']
                captioning_config = (captioning['word_embeddings_file'], captioning['caption_ground_truth'], \
                                     captioning['eval_type'], captioning['caption_search'])
            else:
                captioning_config = None

            self.feeder.add_dataset(dataset_phase, id, path, mean_image, prepend_folder, image_shape, imgproc, raw_image_shape, data_format,
                            frame_format, batch_item, self.num_classes, tag, read_tries, captioning_config)

    # should resume, if set and non-empty
    def should_resume(self):
        return self.resume_file is not None and self.resume_file

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
            error(".ini files deprecated.")

        with open(init_file,"r") as f:
            config = yaml.load(f)[tag_to_read]
            self.read_config(config, init_file)


        print("Initialized run [%s] from file %s" % (self.run_id, init_file))
        sys.stdout.flush()


    def configure_logging(self):
        # configure the logs
        self.timestamp = get_datetime_str()
        logfile = os.path.join(self.run_folder, "log_" + self.run_id + "_" + self.timestamp + ".log")
        self.logger = CustomLogger()
        CustomLogger.instance = self.logger
        self.logger.configure_logging(logfile, self.logging_level, self.email_notify)
        sys.stdout.flush(), sys.stderr.flush()

    # initialize stuff
    def initialize(self, init_file):

        self.initialize_from_file(init_file)
        info("Initialized from configuration file: [%s]" % init_file)

        # if config file is not in the run folder, copy it there to preserve a settings log
        if not (os.path.dirname(init_file) == self.run_folder):
            copyfile(init_file,  os.path.join(self.run_folder, os.path.basename(init_file)))


        # train-val mode has become unsupported
        if self.train and self.val:
            error("Cannot specify simultaneous training and validation run, for now.")
        if not (self.train or self.val):
            error("Neither training nor validation is enabled.")

        # set the tensorboard mode-dependent folder
        self.tensorboard_folder = os.path.join(self.run_folder, self.tensorboard_folder, self.phase)


        # initialize datasets
        self.feeder.initialize_datasets()
        self.feeder.set_phase(self.phase)

        # if not resuming, set start folder according to now()
        if  self.should_resume():
            if self.train:
                # load batch and epoch where training left off
                info("Resuming training.")
                # resume training metadata only in training
                self.train.epoch_index, self.global_step =  self.feeder.resume_snap(self.resume_file)
            if self.val:
                info("Evaluating trained network.")
        else:
            if self.train:
                info("Starting training from scratch.")
            if self.val:
                warning("Starting validation-only run with an untrained network.")

        info("Starting run on folder [%s]." % (self.run_folder))
        return self.feeder

    def resume_graph(self):
        if self.should_resume():
            self.feeder.resume_graph(self.resume_file)


