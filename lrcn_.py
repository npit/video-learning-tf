import tensorflow as tf
import pickle
import  math
# models
from models.alexnet import alexnet
from models.audionet import audionet
from models.lstm import lstm
# util
from utils_ import *
from defs_ import *
from tf_util import *


class LRCN:
    # placeholders
    inputData = None
    inputLabels = None
    caption_lengths = None
    workflow = None

    # output data
    logits = None
    accuracy = None

    # internal
    loss = None
    optimizer = None

    logger = None
    item_logits = None
    item_labels = None
    clip_logits = None
    clip_labels = None
    clip_fusion = None

    current_lr = None
    dcnn_model = None
    lstm_model = None

    dcnn_weights_file = None
    ignorable_variable_names = []

    validation_logits_save_counter = 0
    validation_logits_save_interval = None
    run_id = None
    run_folder = None

    # input
    input = []

    # let there be network
    def create(self, settings, summaries):
        # initializations
        if defs.workflows.is_description(settings.workflow) and defs.workflows.is_image(settings.workflow):
            self.item_logits = []
            self.item_labels = []
            self.non_padding_word_idxs = tf.placeholder(tf.int32, (None))
        else:
            # items refer to the primary unit we operate one, i.e. videos or frames
            self.item_logits = np.zeros([0, settings.network.num_classes], np.float32)
            self.item_labels = np.zeros([0, settings.network.num_classes], np.float32)
            # clips refers to image groups that compose a video, for training with clip information
            self.clip_logits = np.zeros([0, settings.network.num_classes], np.float32)
            self.clip_labels = np.zeros([0, settings.network.num_classes], np.float32)

        # define network input
        self.workflow = settings.workflow
        self.timestamp = settings.timestamp
        self.run_id = settings.run_id
        self.run_folder = settings.run_folder
        self.validation_logits_save_interval = settings.val.logits_save_interval if settings.val else None

        # create the workflows
        # Activity recognition
        if self.workflow == defs.workflows.acrec.singleframe:
           self.create_actrec_singleframe(settings)
        elif self.workflow == defs.workflows.acrec.audio:
            self.create_actrec_audio(settings)
        elif self.workflow == defs.workflows.acrec.lstm:
            self.create_actrec_lstm(settings)
        elif self.workflow == defs.workflows.acrec.multi:
            self.create_actrec_dual(settings)
        # Image description
        elif self.workflow == defs.workflows.imgdesc.inputstep:
            self.create_imgdesc_visualinput(settings)
        elif self.workflow == defs.workflows.imgdesc.statebias:
            self.create_imgdesc_statebias(settings)
        elif self.workflow == defs.workflows.imgdesc.inputbias:
            self.create_imgdesc_inputbias(settings)
        # Video description
        elif self.workflow == defs.workflows.videodesc.fused:
            self.create_videodesc_fusion(settings)
        elif self.workflow == defs.workflows.videodesc.encdec:
            self.create_videodesc_encdec(settings)

        else:
            error("Unknown run mode [%s]" % self.workflow)

        # create the training ops
        if settings.train:
            self.create_training(settings, summaries)

    def precompute_learning_rates(self, settings):
        base_lr = settings.train.base_lr
        decay_params = settings.train.lr_decay
        num_batches = settings.get_num_batches()
        total_num_batches = num_batches * settings.train.epochs
        lr_per_batch = []
        if decay_params is None:
            return [base_lr for _ in range(total_num_batches)]
        log_message = "Dropping LR of %2.5f " % base_lr
        lr_drop_offset = 0 if len(tuple(decay_params)) == 4 else decay_params[-1]
        decay_strategy, decay_scheme, decay_freq, decay_factor = tuple(decay_params[:4])

        if decay_strategy == defs.decay.granularity.exp:
            staircase = False
            log_message += "smoothly "
        elif decay_strategy == defs.decay.granularity.staircase:
            staircase = True
            log_message += "jaggedly "
        else:
            error("Undefined decay strategy %s" % decay_strategy)

        if decay_scheme == defs.decay.scheme.interval:
            # reduce every decay_freq batches
            decay_period = decay_freq
            log_message += "every %d step(s) " % decay_period
        elif decay_scheme == defs.decay.scheme.drops:
            # reduce a total of decay_freq times
            decay_period = math.ceil(total_num_batches / decay_freq)
            log_message += "every ceil[(%d batches x %d epochs) / %d total steps] = %d steps" % \
                (num_batches, settings.train.epochs, decay_freq, decay_period)
        else:
            error("Undefined decay scheme %s" % decay_scheme)

        idx = 0
        while len(lr_per_batch) < total_num_batches:
            if staircase:
                fraction = idx // decay_freq
            else:
                fraction = idx / decay_freq
            current_lr = base_lr * pow(decay_factor,fraction)
            idx = idx + decay_freq
            lr_per_batch.extend([current_lr for _ in range(decay_period)])

        lr_per_batch = lr_per_batch[:total_num_batches]
        log_message += ", mid / last lr is: %1.5f, %1.5f, total drops: %d" % (lr_per_batch[len(lr_per_batch)//2], lr_per_batch[-1], len(lr_per_batch))
        if lr_drop_offset:
            lr_per_batch = [base_lr for _ in range(lr_drop_offset)] + lr_per_batch[0:-lr_drop_offset]
            log_message += " - with a %d-step offset " % lr_drop_offset

        lr_schedule_file = os.path.join(settings.run_folder,settings.run_id + "_lr_decay_schedule.txt")
        with open(lr_schedule_file,"w") as f:
            batches = [ x for _ in range(settings.train.epochs) for x in range(num_batches)]
            if len(batches) != total_num_batches:
                error("Batch length precomputation mismatch")
            epochs = [ep for ep in range(settings.train.epochs) for _ in range(settings.get_num_batches())]
            batches_lr = list(zip(epochs, batches, lr_per_batch))

            for b in batches_lr:
                f.write("Epoch %d/%d, batch %d/%d, lr %2.8f\n" % (b[0]+1, settings.train.epochs,b[1]+1, num_batches,b[2]))
        info(log_message)
        return lr_per_batch

    # training ops
    def create_training(self, settings, summaries):
        info("Creating training: { %s }" % settings.get_train_str())

        self.logits = print_tensor(self.logits, "training: logits : ")
        # configure loss
        with tf.name_scope("cross_entropy_loss"):
            loss_per_vid = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputLabels, name="loss")
            with tf.name_scope('total'):
                self.loss = tf.reduce_mean(loss_per_vid)
            summaries.train.append(add_descriptive_summary(self.loss))

        # configure the learning rate
        learning_rates = self.precompute_learning_rates(settings)
        self.learning_rates = tf.constant(learning_rates,tf.float32,name="Learning_rates")
        self.global_step = tf.Variable(settings.global_step, dtype = tf.int32, trainable=False,name="global_step")
        with tf.name_scope("lr"):
            self.current_lr = self.learning_rates[self.global_step ]
            summaries.train.append(add_descriptive_summary(self.current_lr))

        # setup the training ops, with a potential lr per-layer variation
        if settings.train.lr_mult is not None:
            self.create_multi_tier_learning(settings, summaries)
        else:
            self.create_single_tier_learning(settings, summaries)

        # accuracies
        with tf.name_scope('training_accuracy'):
            with tf.name_scope('correct_prediction_train'):
                # ok for this argmax we gotta squash the labels down to video level.
                correct_predictionTrain = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.inputLabels, 1))
            with tf.name_scope('accuracy_train'):
                self.accuracyTrain = tf.reduce_mean(tf.cast(correct_predictionTrain, tf.float32))

        summaries.train.append(tf.summary.scalar('accuracyTrain', self.accuracyTrain))

    # establish the training ops with slow and fast learning parameters
    def create_multi_tier_learning(self, settings, summaries):
        with tf.name_scope("two_tier_optimizer"):
            # split tensors to slow and fast learning, as per their definition
            regular_vars, modified_vars = [], []
            if self.dcnn_model is not None:
                regular_vars.extend(self.dcnn_model.train_regular)
                modified_vars.extend(self.dcnn_model.train_modified)
            if self.lstm_model is not None:
                regular_vars.extend(self.lstm_model.train_regular)
                modified_vars.extend(self.lstm_model.train_modified)
            info("Setting up two-tier training with a factor of %f for the %d layer(s): %s" % (
            settings.train.lr_mult, len(modified_vars), [ m.name for m in modified_vars]))

            # setup the two optimizer
            if settings.train.optimizer == defs.optim.sgd:
                opt = tf.train.GradientDescentOptimizer(self.current_lr, name="sgd_base")
            elif settings.train.optimizer == defs.optim.adam:
                opt = tf.train.AdamOptimizer(self.current_lr)
            else:
                error("Undefined optimizer %s" % settings.train.optimizer)

            # computer two-tier grads
            grads = opt.compute_gradients(self.loss, var_list=regular_vars)
            if settings.train.clip_grads is not None:
                clipmin, clipmax = settings.train.clip_grads
                grads = [(tf.clip_by_global_norm(grad, clipmax), var) for grad, var in grads]
            trainer_base = opt.apply_gradients(grads)

            modified_lr = self.current_lr * settings.train.lr_mult
            opt_mod = tf.train.GradientDescentOptimizer(modified_lr, name="sgd_mod")
            grads_mod = opt_mod.compute_gradients(self.loss, var_list=modified_vars)
            if settings.train.clip_grads is not None:
                clipmin, clipmax = settings.train.clip_grads
                grads_mod = [(tf.clip_by_global_norm(grad_mod, clipmax), var_mod) for grad_mod, var_mod in
                             grads_mod]
            trainer_modified = opt.apply_gradients(grads_mod, global_step=self.global_step)



            self.optimizer = tf.group(trainer_base, trainer_modified)
        with tf.name_scope("grads_norm"):
            grads_norm = tf.reduce_mean(list(map(tf.norm, grads)))
            summaries.train.append(add_descriptive_summary(grads_norm))
        with tf.name_scope("grads_mod_norm"):
            grads_mod_norm = tf.reduce_mean(list(map(tf.norm, grads_mod)))
            summaries.train.append(add_descriptive_summary(grads_mod_norm))

    def create_single_tier_learning(self, settings, summaries):
        # single lr for all
        info("Setting up training with a global learning rate.")
        with tf.name_scope("single_tier_optimizer"):
            if settings.train.optimizer == defs.optim.sgd:
                opt = tf.train.GradientDescentOptimizer(self.current_lr)
            elif settings.train.optimizer == defs.optim.adam:
                opt = tf.train.AdamOptimizer(self.current_lr)
            else:
                error("Undefined optimizer %s" % settings.train.optimizer)

            grads_vars = opt.compute_gradients(self.loss)
            grads = [grad for grad, _ in grads_vars]
            if settings.train.clip_norm:
                #clipmin, clipmax = settings.clip_grads
                #max_norm = settings.clip_norm
                grads, _ = tf.clip_by_global_norm(grads, settings.train.clip_norm)
                grads_vars = zip(grads, [v for _, v in grads_vars])
            self.optimizer = opt.apply_gradients(grads_vars, global_step=self.global_step)

        with tf.name_scope('grads_norm'):
            grads_norm = tf.reduce_mean(list(map(tf.norm, grads)))
            grads_norm = print_tensor(grads_norm, "grads_clipped" )
            summaries.train.append(add_descriptive_summary(grads_norm))

    # workflows

    # Activity recognition
    def create_actrec_singleframe(self, settings, inputData = None, inputLabels = None):
        if inputData is None:
            info("Dcnn workflow [%s][%s][%s]" % (settings.network.frame_fusion_type, settings.network.frame_fusion_method, settings.network.frame_encoding_layer))
            # define label inputs
            self.inputLabels = tf.placeholder(tf.int32, [None, settings.network.num_classes], name="input_labels")
            self.inputData = tf.placeholder(tf.float32, (None,) + settings.network.image_shape, name='input_frames')
            self.input.append((self.inputData, defs.net_input.visual, defs.dataset_tag.main))
            self.input.append((self.inputLabels, defs.net_input.labels, defs.dataset_tag.main))
        else:
            info("(cont.) Dcnn workflow [%s]" % str(settings.network.frame_fusion_type))
            self.inputData = inputData
            self.inputLabels = inputLabels

        # get fpc from main dataset
        fpc = settings.get_dataset_by_tag(defs.dataset_tag.main)[0].num_frames_per_clip

        # create the singleframe workflow
        with tf.name_scope("dcnn_workflow"):
            # encode the frames
            encoding_layer = None
            if settings.network.frame_fusion_type == defs.fusion_type.early:
                encoding_layer = settings.network.frame_encoding_layer
            encoded_frames = self.make_dcnn(settings, encoding_layer)
            encoded_dim = int(encoded_frames.shape[-1])
            encoded_frames = print_tensor(encoded_frames, "dcnn output")
            encoded_frames = tf.reshape(encoded_frames, (-1, fpc, encoded_dim),
                                     name="reshape_dcnn_out")
            debug("reshaped dcnn output: %s" % str(encoded_frames))
            if settings.network.frame_fusion_type == defs.fusion_type.late:
                # if we have no clips, we're done
                if settings.get_datasets()[0].input_mode == defs.input_mode.image or fpc == 1:
                    return
                if encoded_dim != settings.network.num_classes:
                    error("Specified late fusion but dcnn output is %d and num. classes is %d" % (encoded_dim, settings.network.num_classes))

                self.logits = apply_temporal_fusion(encoded_frames, settings.network.num_classes, fpc,
                                                     settings.network.frame_fusion_method)

            elif settings.network.frame_fusion_type == defs.fusion_type.early:
                # frames - encode to vectors - pool to clips - logit per clip
                clip_vectors = apply_temporal_fusion(encoded_frames, encoded_dim, fpc,
                                             settings.network.frame_fusion_method, lstm_encoder=None)
                # classify to the desired dimension
                self.logits = convert_dim_fc(clip_vectors, settings.network.num_classes)
            else:
                # no frame fusion specified
                if fpc == 1:
                    self.logits = encoded_frames
                else:
                    error("Dataset has non-unitary fpc, but no frame fusion has been specified.")

        info("logits out : [%s]" % self.logits.shape)

    def create_actrec_lstm(self, settings, inputData = None, inputLabels = None):
        # define label inputs
        if inputData is None:
            settings.frame_encoding_layer = None
            self.inputLabels = tf.placeholder(tf.int32, [None, settings.network.num_classes], name="input_labels")
            self.inputData = tf.placeholder(tf.float32, (None,) + settings.network.image_shape, name='input_frames')
            self.input.append((self.inputData, defs.net_input.visual, defs.dataset_tag.main))
            self.input.append((self.inputLabels, defs.net_input.labels, defs.dataset_tag.main))
        else:
            self.inputData = inputData
            self.inputLabels = inputLabels

        # create the lstm workflow
        with tf.name_scope("lstm_workflow"):
            if settings.input_mode != defs.input_mode.video:
                error("The LSTM workflow only available for video input mode")
            if settings.network.frame_fusion_type != defs.fusion_type.none:
                error("The LSTM workflow is only compatible with frame fusion mode %s. Specify lstm fusion in the network.lstm_params" % defs.fusion_type.none)

            # create vectorial input, if not supplied
            if inputData is None:
                # DCNN for frame encoding
                encodedFrames = self.make_dcnn(settings)
            else:
                encodedFrames = inputData

            # LSTM for frame sequence classificationfor frame encoding
            self.lstm_model = lstm.lstm()
            input_dim = int(encodedFrames.shape[1])
            dropout = settings.train.dropout_keep_prob if settings.train else 0.0
            self.logits, output_state = self.lstm_model.forward_pass_sequence(encodedFrames, None, input_dim, settings.network.lstm_params,
                                settings.network.num_classes, settings.get_datasets()[0].num_frames_per_clip, None, dropout)
            if settings.network.lstm_params[2] == defs.fusion_method.state:
                # select the last state vector
                output_state = output_state[-1].h
                if int(output_state.shape[1]) != settings.network.num_classes:
                    info("Fused lstm input with %s method, but num hidden is %d and num classes is %d - adding fc layer"  \
                          %(defs.fusion_method.state, settings.network.lstm_params[0], settings.network.num_classes))
                    self.logits = convert_dim_fc(output_state, settings.network.num_classes)
                else:
                    self.logits = output_state

            info("logits : [%s]" % self.logits.shape)

    def create_actrec_audio(self, settings):
        # define label inputs
        #batchLabelsShape = [None, settings.network.num_classes]
        #self.inputLabels = tf.placeholder(tf.int32, batchLabelsShape, name="input_labels")

        #settings.frame_encoding_layer = None
        ## create the singleframe workflow
        #with tf.name_scope("audionet_workflow"):
        #    info("Audionet workflow")
        #    # single DCNN, classifying individual frames
        #    self.dcnn_model = audionet.audionet()
        #    self.dcnn_model.create(dataset.image_shape, settings.network.num_classes)
        #    self.inputData, self.logits = self.dcnn_model.get_io()
        ## reshape to num_items x num_frames_per_item x dimension
        #self.logits = tf.reshape(self.logits, (-1, settings.get_datasets()[0].num_frames_per_clip, settings.network.num_classes),
        #                         name="reshape_framelogits_pervideo")

        #if dataset.input_mode == defs.input_mode.image or settings.get_datasets()[0].num_frames_per_clip == 1:
        #    return

        # fuse the logits on the temporal dimension
        #self.logits = apply_temporal_fusion(self.logits, settings.network.num_classes, settings.get_datasets()[0].num_frames_per_clip, settings.network.frame_fusion_method)

        #info("logits out : [%s]" % self.logits.shape)
        pass

    def create_actrec_dual(self, settings):
        # workflow that encodes images from two datasets to vectors, which are fused as specififed by the
        # dataset and frame fusion parameters. The classification itself is determined by the multi_workflow network
        # parameter


        info("%s workflow type: [%s]" % (settings.workflow, settings.network.multi_workflow))

        # a dcnn for each dataset, mapping an image to a vector.
        self.inputLabels = tf.placeholder(tf.int32, [None, settings.network.num_classes], name="input_labels")
        self.input.append((self.inputLabels, defs.net_input.labels, defs.dataset_tag.main))

        if type(settings.network.frame_encoding_layer) != list:
            settings.network.frame_encoding_layer = [settings.network.frame_encoding_layer for _ in range(2)]

        enc_layer_main, enc_layer_aux = settings.network.frame_encoding_layer
        # make a dcnn for the main input
        ids_main = ",".join([d.id for d in settings.get_dataset_by_tag(defs.dataset_tag.main)])
        info("Dcnn [%s]-[%s]-[%s]" % (defs.dataset_tag.main, ids_main, enc_layer_main))
        input1 = tf.placeholder(tf.float32, (None,) + settings.network.image_shape, name='input_frames1')
        self.input.append((input1, defs.net_input.visual, defs.dataset_tag.main))
        self.inputData = input1
        encoded1, dcnn1 = self.make_dcnn(settings, output_layer=enc_layer_main, get_model = True)
        dim1 = int(encoded1.shape[-1])
        fpc1 = settings.get_dataset_by_tag(defs.dataset_tag.main)[0].num_frames_per_clip

        # make dcnn for auxiliary input
        ids_aux = ",".join([d.id for d in settings.get_dataset_by_tag(defs.dataset_tag.aux)])
        info("Dcnn [%s]-[%s]-[%s]" % (defs.dataset_tag.aux, ids_aux, enc_layer_aux))
        input2 = tf.placeholder(tf.float32, (None,) + settings.network.image_shape, name='input_frames2')
        self.input.append((input2, defs.net_input.visual, defs.dataset_tag.aux))
        self.inputData = input2
        encoded2, dcnn2 = self.make_dcnn(settings, output_layer=enc_layer_aux, get_model = True)
        dim2 = int(encoded2.shape[-1])
        fpc2 = settings.get_dataset_by_tag(defs.dataset_tag.aux)[0].num_frames_per_clip


        # early / late frame fusion: framefusion before / after the classification
        # early / late frame fusion: framefusion before / after the frame fusion, if the latter is early. Always b4 classification
        fusion_order = []
        # early dataset fusion: before frame fusion
        if settings.network.dataset_fusion_type == defs.fusion_type.early:
            fusion_order.append("dataset")
        # early frame fusion: before classification
        if settings.network.frame_fusion_type == defs.fusion_type.early:
            fusion_order.append("frames")
        # late dataset fusion: after frame fusion
        if settings.network.dataset_fusion_type == defs.fusion_type.late:
            fusion_order.append("dataset")


        for i, fusion in enumerate(fusion_order):
            if fusion == "dataset":
                if i == 0 and fpc1 != fpc2:
                    error("Attempted early dataset fusion with different fpcs: %d and %d for %s and %s." % (fpc1, fpc2, ids_main, ids_aux))
                # dataset before frames
                if settings.network.dataset_fusion_method == defs.fusion_method.avg:
                    if dim1 != dim2:
                        error("%s dataset fusion requires same vector dim, but it's %d and %d for %s and %s." % (settings.network.dataset_fusion_method, dim1, dim2, ids_main, ids_aux))
                    conc = tf.stack([encoded1, encoded2])
                    fused = tf.reduce_mean(conc,axis=0)
                elif settings.network.dataset_fusion_method == defs.fusion_method.concat:
                    fused = tf.concat([encoded1, encoded2],axis=1)
                else:
                    error("Only %s and %s fusion methods are meaningfull for the %s workflow" % \
                          (defs.fusion_method.avg, defs.fusion_method.concat, settings.workflow))
                dataset_fused_dim = int(fused.shape[-1])
                info("Fused dataset vectors via %s. Result: %s" % (settings.network.dataset_fusion_method, str(fused.shape)))
            elif fusion == "frames":
                if i > 0 and "dataset" in fusion_order[:i]:
                    # already fused dsets
                    fused = tf.reshape(fused, [ -1, fpc1, dataset_fused_dim])
                    fused = apply_temporal_fusion(fused, dataset_fused_dim, fpc1, settings.network.frame_fusion_method)
                    info("Early-late fused frames per clip via %s: %s" % (settings.network.frame_fusion_method, str(fused.shape)))
                else:
                    # fuse the video frames of the main dataset
                    encoded1 = tf.reshape(encoded1, ( -1, fpc1, dim1))
                    encoded1 = apply_temporal_fusion(encoded1, dim1, fpc1, settings.network.frame_fusion_method)
                    info("%s: Early-early fused frames per clip via %s: %s" % (ids_main, settings.network.frame_fusion_method, str(encoded1.shape)))
                    # fuse the other - aux
                    encoded2 = tf.reshape(encoded2, [ -1, fpc2, dim2])
                    encoded2 = apply_temporal_fusion(encoded2, dim2, fpc2, settings.network.frame_fusion_method)
                    info("%s: Early-early fused frames per clip via %s: %s" % (ids_aux, settings.network.frame_fusion_method, str(encoded2.shape)))

        # classify
        multi_workflow = settings.network.multi_workflow
        if multi_workflow == defs.workflows.multi.fc:
            # data has been fused across frames and datasets - clip vectors remain.
            # classify with an fc layer
            info("Using multi workflow: [%s]" % multi_workflow)
            self.logits = convert_dim_fc(fused, settings.network.num_classes)
            if "frames" not in fusion_order:
                # if no frame fusion specified: better be applicable
                if settings.network.frame_fusion_type == defs.fusion_type.none and fpc1 != 1:
                    error("Did not specify frame fusion method in [%s] workflow and fpc is %d" % (multi_workflow, fpc1))
                # it's gotta be late
                elif settings.network.frame_fusion_type != defs.fusion_type.late:
                    error("Arrived at multi fc workflow with unfused frames, fusion type is %s." % settings.network.frame_fusion_type)

                # apply late fusion, if not already (i.e. early dataset fusion)
                self.logits = print_tensor(self.logits, "raw fc logits")
                self.logits = tf.reshape(self.logits,[-1, fpc1, settings.network.num_classes])
                self.logits = apply_temporal_fusion(self.logits, dataset_fused_dim, fpc1,settings.network.frame_fusion_method)

        elif multi_workflow == defs.workflows.multi.lstm:
            # these workflows operate on data with no fused frames
            # data has been fused across datasets - can classify
            if settings.network.frame_fusion_type != defs.fusion_type.none:
                error("[%s] fused workflow requires frame fusion type to be [%s]" % (multi_workflow, defs.fusion_type.none))
            self.create_actrec_lstm(settings, inputData = fused, inputLabels = self.inputLabels)
                #self.lstm_model = lstm.lstm()
                #dataset_fused_dim = int(fused.shape[-1])
                #self.logits, _ = self.lstm_model.forward_pass_sequence(fused, None, dataset_fused_dim , settings.network.lstm_num_layers,
                #                            settings.network.lstm_num_hidden, settings.network.num_classes, fpc1, None,
                #                            settings.network.frame_fusion_method, settings.train.dropout_keep_prob)
        else:
            # these workflows require only *one* of the datasets to be frame-fused
            # the dataset to be fused will be the one with the aux tag
            # fuse the aux dataset
            encoded2 = tf.reshape(encoded2, [ -1, fpc2, dim2])
            fused_dset = apply_temporal_fusion(encoded2, dim2, fpc2, settings.network.frame_fusion_method)
            seq_dset = encoded1
            fpc = fpc1
            debug("Early fused frames per clip via %s: %s" % (settings.network.frame_fusion_method, str(encoded2.shape)))

            seq_dim = int(seq_dset.shape[-1])
            fused_dim = int(fused_dset.shape[-1])

            # supply the non-fused dataset to an lstm input, and the fused one as a bias / concat type
            if multi_workflow == defs.workflows.multi.lstm_conc:
                # concat the fused clip vectors to each framevector
                enc_concatted = vec_seq_concat(seq_dset, fused_dset, fpc)
                enc_concatted = print_tensor(enc_concatted, "lstm-concatted tensor")
                # proceed with regular lstm
                self.create_actrec_lstm(settings, inputData = enc_concatted, inputLabels = self.inputLabels)
            elif multi_workflow == defs.workflows.multi.lstm_sbias:
                # use the fused clip vectors as a state bias
                lstm_model = lstm.lstm()
                self.logits, _ = lstm_model.forward_pass_sequence(seq_dset, fused_dset, seq_dim, settings.network.lstm_params,
                                                settings.network.num_classes, fpc, None, settings.train.dropout_keep_prob)

            elif multi_workflow == defs.workflows.multi.lstm_ibias:
                # use the fused clip vectors as an additional input at the first step
                if seq_dim != fused_dim:
                    error("%s workflow requires same encoded dimension for both datasets. Instead it is %d, %d for fused and seq." %  \
                          (multi_workflow, fused_dim, seq_dim))

                batch_size = tf.shape(fused_dset)[0]
                # reshape the fused dataset to batch_size x fpc=1 x dim
                reshaped_fused_dset = tf.reshape(fused_dset, [batch_size, 1, fused_dim])
                # reshape the images accordingly
                reshaped_seq_dset= tf.reshape(seq_dset, [batch_size, fpc, seq_dim])
                # insert the fused as the first item in the seq - may need tf.expand on the fused
                input_biased_seq = tf.concat([reshaped_fused_dset, reshaped_seq_dset], axis=1)
                # increase the seq len to account for the input bias extra timestep
                augmented_fpc = fpc + 1
                info("Input bias augmented fpc: %d + 1 = %d" % (fpc, augmented_fpc))
                # restore to batchsize*seqlen x embedding_dim
                input_biased_seq = tf.reshape(input_biased_seq ,[augmented_fpc * batch_size, seq_dim])
                # classify
                lstm_model = lstm.lstm()
                self.logits, _ = lstm_model.forward_pass_sequence(input_biased_seq, None, seq_dim, settings.network.lstm_params,
                                                               settings.network.num_classes, augmented_fpc, None,
                                                               settings.train.dropout_keep_prob)

        info("Logits: %s" % str(self.logits.shape))





    # Image description
    def create_imgdesc_visualinput(self, settings, dataset):

        # make sure input mode is image
        if dataset.input_mode != defs.input_mode.image:
            error("The image description workflow works only in image input mode.")
        with tf.name_scope("imgdesc_workflow"):
            self.make_description_placeholders(settings,dataset)
            self.inputData, encodedFrames = self.make_dcnn(dataset,settings)

            self.lstm_model = lstm.lstm()
            visual_dim, embedding_dim = int(encodedFrames.shape[-1]), int(dataset.embedding_matrix.shape[-1])
            if settings.train:
                # repeat the images in the batch seqlength times horizontally (batchsize x (dim * seq))
                encodedFrames = tf.tile(encodedFrames, [1, dataset.max_sequence_length])
                # restore to 'one image per column', i.e. (batchsize * seq x dim)
                encodedFrames = tf.reshape(encodedFrames, [-1, visual_dim], name="restore_to_sequence")
                debug("Duplicated encoded frames : [%s]" % encodedFrames.shape)
                # horizontal concat the images to the words
                frames_words = tf.concat([encodedFrames, self.word_embeddings], axis=1)

                self.logits, _ = self.lstm_model.forward_pass_sequence(frames_words, None, embedding_dim + visual_dim, settings.network.lstm_num_layers,
                                                      settings.network.lstm_num_hidden, settings.network.num_classes, dataset.max_sequence_length,
                                                      self.caption_lengths, defs.fusion.reshape, settings.train.dropout_keep_prob)

                # remove the logits corresponding to padding
                self.logits = tf.gather(self.logits, self.non_padding_word_idxs)
                # re-merge the tensor list into a tensor
                self.logits = tf.concat(self.logits, axis=0)
                debug("final filtered logits : [%s]" % self.logits.shape)

            else:
                vinput_mode = defs.rnn_visual_mode.input_concat
                self.logits = self.lstm_model.generate_feedback_sequence(encodedFrames, dataset.batch_size_val, settings.network.num_classes,
                                                                         dataset.max_sequence_length, settings.network.lstm_num_hidden, settings.network.lstm_num_layers,
                                                                         dataset.embedding_matrix[dataset.vocabulary.index("BOS"), :], dataset.embedding_matrix, vinput_mode)
                self.logits = tf.reshape(self.logits,[-1, dataset.max_sequence_length])

    def create_imgdesc_inputbias(self, settings, dataset):
        # the implementation here implements the "show and tell model"
        # make sure input mode is image
        if dataset.input_mode != defs.input_mode.image:
            error("The image description workflow works only in image input mode.")
        with tf.name_scope("imgdesc_workflow"):
            self.make_description_placeholders(settings, dataset)
            self.inputData, encodedFrames = self.make_dcnn(dataset, settings)
            # make recurrent network
            self.lstm_model = lstm.lstm()
            # map the images to the embedding input
            encodedFrames = convert_dim_fc(encodedFrames, dataset.get_embedding_dim(), "visual_embedding_fc")

            if settings.train:
                batch_size_train = tf.shape(encodedFrames)[0]
                # reshape the embeddings to batch_size x seq_len x embedding_dim
                reshaped_word_embeddings = tf.reshape(self.word_embeddings,
                                                      [batch_size_train,
                                                       dataset.max_sequence_length,
                                                       dataset.get_embedding_dim()])
                # reshape the images accordingly
                encodedFrames = tf.reshape(encodedFrames, [batch_size_train, 1, dataset.get_embedding_dim()])
                # insert the images as the first item in the embeddings - may need tf.expand on the images
                input_biased_embeddings = tf.concat([encodedFrames, reshaped_word_embeddings], axis=1)
                # increase the seq len to account for Wthe input bias extra timestep
                augmented_sequence_length = dataset.max_sequence_length + 1
                # restore to batchsize*seqlen x embedding_dim
                input_biased_embeddings = tf.reshape(input_biased_embeddings,[augmented_sequence_length * batch_size_train,
                                                                              dataset.get_embedding_dim()])

                info("Incrementing sequence length to %d for the input bias step" % (augmented_sequence_length))
                self.logits, _ = self.lstm_model.forward_pass_sequence(input_biased_embeddings, None,
                                                                       int(dataset.embedding_matrix.shape[1]),
                                                                       settings.network.lstm_num_layers,
                                                                       settings.network.lstm_num_hidden, settings.network.num_classes,
                                                                       augmented_sequence_length,
                                                                       self.caption_lengths, defs.fusion_method.reshape,
                                                                       settings.train.dropout_keep_prob)
                # remove the logits corresponding to padding
                self.logits = tf.gather(self.logits, self.non_padding_word_idxs)
                # re-merge the tensor list into a tensor
                self.logits = tf.concat(self.logits, axis=0)
                debug("final filtered logits : [%s]" % self.logits.shape)

            else:
                # increase the seq len to account for the input bias
                augmented_sequence_length = dataset.max_sequence_length + 1
                info("Incrementing sequence length to %d for the input bias step" % (augmented_sequence_length))
                vinput_mode = defs.rnn_visual_mode.input_bias
                self.logits = self.lstm_model.generate_feedback_sequence(encodedFrames, dataset.batch_size_val,
                                                                         settings.network.num_classes,
                                                                         augmented_sequence_length,
                                                                         settings.network.lstm_num_hidden,
                                                                         settings.network.lstm_num_layers,
                                                                         dataset.embedding_matrix[
                                                                         dataset.vocabulary.index("BOS"), :],
                                                                         dataset.embedding_matrix, vinput_mode)
                debug("Raw logits : [%s]",str(self.logits.shape))
                # output is a sequence length * batchsize vector
                self.logits = tf.reshape(self.logits, [-1, dataset.max_sequence_length])

    def create_imgdesc_statebias(self, settings, dataset):
        # the implementation here is similar to the "show and tell model"
        # make sure input mode is image
        if dataset.input_mode != defs.input_mode.image:
            error("The image description workflow works only in image input mode.")
        with tf.name_scope("imgdesc_workflow"):
            self.make_description_placeholders(settings,dataset)
            self.inputData, encodedFrames = self.make_dcnn(dataset,settings)
            # make recurrent network
            self.lstm_model = lstm.lstm()
            if settings.train:
                self.logits, _ = self.lstm_model.forward_pass_sequence(self.word_embeddings, encodedFrames, int(dataset.embedding_matrix.shape[1]),
                                                      settings.network.lstm_num_layers, settings.network.lstm_num_hidden, settings.network.num_classes,
                                                      dataset.max_sequence_length, self.caption_lengths, defs.fusion_method.reshape, settings.train.dropout_keep_prob )
                # remove the logits corresponding to padding
                self.logits = tf.gather(self.logits, self.non_padding_word_idxs)
                # re-merge the tensor list into a tensor
                self.logits = tf.concat(self.logits, axis=0)
                debug("final filtered logits : [%s]" % self.logits.shape)

            else:
                vinput_mode = defs.rnn_visual_mode.state_bias
                self.logits = self.lstm_model.generate_feedback_sequence(encodedFrames, dataset.batch_size_val,
                                                           settings.network.num_classes, dataset.max_sequence_length, settings.network.lstm_num_hidden,
                                                           settings.network.lstm_num_layers, dataset.embedding_matrix[dataset.vocabulary.index("BOS"),:],
                                                           dataset.embedding_matrix, vinput_mode)
                # output is a sequence length * batchsize vector
                self.logits = tf.reshape(self.logits,[-1, dataset.max_sequence_length])


    # make description placeholders
    def make_description_placeholders(self, settings, dataset):
        # set up placeholders
        self.caption_lengths = tf.placeholder(tf.int32, shape=(None), name="words_per_item")
        self.inputLabels = tf.placeholder(tf.int32, [None, settings.network.num_classes], name="input_labels")
        self.word_embeddings = tf.placeholder(tf.float32, shape=(None, dataset.embedding_matrix.shape[1]),
                                              name="word_embeddings")
        debug("input labels : [%s]" % self.inputLabels)

    # make then dcnn network
    def make_dcnn(self, settings, output_layer = None, get_model = False):

        # make sure dcnn weights are good2go
        self.dcnn_weights_file = os.path.join(os.getcwd(), "models/alexnet/bvlc_alexnet.npy")
        if not os.path.exists(self.dcnn_weights_file):
            error("Weights file %s does not exist." % self.dcnn_weights_file)

        # DCNN for frame encoding
        model = alexnet.dcnn()
        # get the first dataset through the DCNN
        model.create(self.inputData, self.dcnn_weights_file, settings.network.num_classes, output_layer, settings.network.load_weights)
        outputData = model.get_output()
        debug("dcnn input : [%s]" % self.inputData.shape)
        debug("dcnn output : [%s]" % outputData.shape)
        outputData = print_tensor(outputData, "encoded frames")
        if not get_model:
            self.dcnn_model = model
            return outputData
        else:
            return outputData, model

    # video description
    def create_videodesc_fusion(self, settings):
        # Venugopalan et al. 2015 : fuse video frames, do img desc.

        # get dcnn encodings for each frame
        self.inputData, encoded_frames = self.make_dcnn(settings)

        # fuse video frames to a single vector
        if settings.network.frame_fusion_method == defs.fusion_method.avg:
            fused_frames = tf.reduce_mean(encoded_frames, 0)

        self.make_description_placeholders(settings)
        if settings.train:
            self.logits, _ = self.lstm_model.forward_pass_sequence(self.word_embeddings, fused_frames,
                                                                   int(settings.get_datasets().embedding_matrix.shape[1]),
                                                                   settings.network.lstm_num_layers, settings.network.lstm_num_hidden,
                                                                   settings.network.num_classes,
                                                                   settings.dataset.max_sequence_length, self.caption_lengths,
                                                                   defs.fusion_method.reshape, settings.train.dropout_keep_prob)
            # remove the logits corresponding to padding
            self.logits = tf.gather(self.logits, self.non_padding_word_idxs)
            # re-merge the tensor list into a tensor
            self.logits = tf.concat(self.logits, axis=0)
            debug("final filtered logits : [%s]" % self.logits.shape)

    def create_videodesc_encdec(self, settings, dataset):
        # Venugopalan et al. 2016 : lstm encoder - decoder

        # get dcnn encodings for each frame
        self.inputData, encoded_frames = self.make_dcnn(dataset, settings)
        # frame sequence is mapped to a fixed-length vector via an lstm
        encoder = lstm.lstm()
        encoder.define_encoder(encoded_frames, settings, dataset)
        encoded_state = encoder.get_output()
        self.make_description_placeholders(settings, dataset)
        # the rest of the workflow is identical to the image description workflow
        self.make_imgdesc_prepro(settings, dataset, encoded_state)

    def get_ignorable_variable_names(self):
        ignorables = []

        if self.dcnn_model:
            ignorables.extend(self.dcnn_model.ignorable_variable_names)
        if self.lstm_model:
            ignorables.extend(self.lstm_model.ignorable_variable_names)
        if ignorables:
            info("Getting lrcn raw ignorables: %s" % str(ignorables))
        ignorables = [drop_tensor_name_index(s) for s in ignorables]
        return list(set(ignorables))

    # process description logits
    def process_description_validation_logits(self, logits, labels, dataset, fdict, padding):
        caption_lengths = fdict[self.caption_lengths]
        assert (len(logits) - padding == len(caption_lengths)), "Logits, labels length mismatch (%d, %d)" % (len(logits)-padding, len(caption_lengths))
        eos_index = dataset.vocabulary.index("EOS")
        # logits is words
        for idx in range(len(logits) - padding):
            image_logits = logits[idx,:]
            image_logits = image_logits[:dataset.max_caption_length]
            eos_position_binary_idx = [1 if x == eos_index else 0 for x in image_logits]

            if any(eos_position_binary_idx):
                # keep up to but not including eos. Get first index, if multiple .
                first_eos = eos_position_binary_idx.index(1)
                image_logits = image_logits[0:first_eos]
            # else, no EOS exists in the predicted caption
            # append the vector
            self.item_logits.append(image_logits)

        # get the labels. In validation mode, labels are EOS-free.
        cumulative_offset = 0
        for item_idx, cap_len in enumerate(caption_lengths):
            label_idxs = [ x + cumulative_offset for x in list(range(cap_len))]
            item_labels = labels[label_idxs,:]
            self.item_labels.append(item_labels)
            cumulative_offset = cumulative_offset + cap_len

    # validation accuracy computation
    def process_validation_logits(self, tag, settings, logits, fdict, padding):
        labels = fdict[self.inputLabels]
        dataset = settings.get_dataset_by_tag(tag)[0]
        # processing for image description
        if defs.workflows.is_description(self.workflow):
            self.process_description_validation_logits(logits, labels, dataset, fdict, padding)
            return

        # batch item contains logits that correspond to whole clips. Accumulate to clip storage, and check for aggregation.
        if dataset.batch_item == defs.batch_item.clip:
            # per-clip logits in input : append to clip logits accumulator
            self.clip_logits = np.vstack((self.clip_logits, logits))
            self.clip_labels = np.vstack((self.clip_labels, labels))
            debug("Adding %d,%d clip logits and labels to a total of %d,%d." % (
                logits.shape[0], labels.shape[0], self.clip_logits.shape[0], self.clip_labels.shape[0]))

            cpv = dataset.clips_per_video[dataset.video_index]
            # while possible, pop a chunk for the current cpv, aggregate, and add to video logits accumulator
            while dataset.video_index < len(dataset.clips_per_video) and cpv <= len(self.clip_logits):

                # aggregate the logits and add to video logits accumulation
                self.apply_clip_fusion(self.clip_logits, cpv, self.clip_labels)
                # delete them from the accumulation
                self.clip_logits = self.clip_logits[cpv:,:]
                self.clip_labels = self.clip_labels[cpv:,:]

                debug("Aggregated %d clips to the %d-th video. Video accumulation is now %d,%d - clip accumulation is %d, %d." %
                                  (dataset.clips_per_video[dataset.video_index], 1 + dataset.video_index, len(self.item_logits),
                                   len(self.item_labels), len(self.clip_logits), len(self.clip_labels)))
                # advance video index
                dataset.video_index = dataset.video_index + 1
                if dataset.video_index >= len(dataset.clips_per_video):
                    break
                cpv = dataset.clips_per_video[dataset.video_index]
        else:
            # batch items are whole items of data
            if dataset.input_mode == defs.input_mode.video:
                # can directly fuse and append to video accumulators
                maxvid = dataset.batch_index * dataset.batch_size
                minvid = maxvid - dataset.batch_size

                for vidx in range(minvid, maxvid):
                    if vidx >= dataset.num_items:
                        break
                    cpv = dataset.clips_per_video[vidx]
                    debug("Aggregating %d clips for video %d in video batch mode" % (cpv, vidx + 1))
                    self.apply_clip_fusion(logits, cpv, labels, settings.network.clip_fusion_method)
                    logits = logits[cpv:,:]
                    labels = labels[cpv:,:]
                if not (len(logits) == 0 and len(labels) == 0):
                    error("Logits and/or labels non empty at the end of video item mode aggregation!")
                debug("Video logits and labels accumulation is now %d,%d video in video batch mode." %
                                  (len(self.item_logits), len(self.item_labels)))
            else:
                # frames, simply append
                self.add_item_logits_labels(logits,labels)

    def save_validation_logits_chunk(self, save_all = False):
        # if saving is not enabled or no logits are stored, leave
        if self.validation_logits_save_interval is None or len(self.item_logits) == 0:
            return
        # if logits saving is set to once at the end
        if self.validation_logits_save_interval <= 0:
            # if we are at the end, save
            if save_all:
                # all extracted logits are in the container
                save_file = os.path.join(self.run_folder,"validation_logits_%s_%s.total" % (self.run_id, self.timestamp))
                info("Saving all %d extracted validation logits to %s" % (len(self.item_logits), save_file))
                with open(save_file, "wb") as ff:
                    pickle.dump(self.item_logits, ff)
            # else, just return
            return

        # if logits saving is done in batches, save either if batch is full or if it is the last step
        if len(self.item_logits) >= self.validation_logits_save_interval or save_all:
            save_file = os.path.join(self.run_folder,"validation_logits_%s_%s.part_%d" %
                                     ( self.run_id, self.timestamp, self.validation_logits_save_counter))
            info("Saving a %d-sized chunk of validation logits to %s" % (len(self.item_logits), save_file))
            with open(save_file, "wb") as f:
                pickle.dump(self.item_logits, f)
            # reset the container
            if type(self.item_logits) == np.ndarray:
                num_classes = int(self.item_logits.shape[-1])
                del self.item_logits
                self.item_logits = np.zeros([0, num_classes], np.float32)
            else:
                # list
                del self.item_logits
                self.item_logits = []

            self.validation_logits_save_counter += 1

    def load_validation_logits_chunk(self, chunk_idx):
        if self.validation_logits_save_interval is None:
            return self.item_logits
        save_file = os.path.join(self.run_folder,"validation_logits_%s_%s.part_%d" % ( self.run_id, self.timestamp, chunk_idx))
        with open(save_file, "rb") as f:
            logits_chunk = pickle.load(f)
        return logits_chunk

    def apply_clip_fusion(self, clips_logits, cpv, video_labels, clip_fusion):
        curr_clips = clips_logits[0:cpv,:]
        video_label = video_labels[0,:]
        if clip_fusion == defs.fusion_method.avg:
            video_logits = np.mean(curr_clips, axis=0)
        elif clip_fusion == defs.fusion_method.last:
            video_logits = curr_clips[-1, :]
            #elif clip_fusion == defs.fusion_method.
        # add logits, label to the video accumulation
        self.add_item_logits_labels(video_logits, video_label)

    def add_item_logits_labels(self,logits,label):
        # add logits, label to the video accumulation
        self.item_logits = np.vstack((self.item_logits, logits))
        self.item_labels = np.vstack((self.item_labels, label))

    def get_accuracy(self):
        # compute accuracy
        info("Computing accuracy")
        accuracies = []
        curr_item_idx = 0
        # compute partial accuracies for each saved chunk
        for saved_idx in range(self.validation_logits_save_counter):
            logits = self.load_validation_logits_chunk(saved_idx)
            chunk_size = len(logits)
            labels = self.item_labels[curr_item_idx:curr_item_idx + chunk_size, :]
            accuracies.append(self.get_chunk_accuracy(logits, labels))
            curr_item_idx += chunk_size
            info("Processed saved chunk %d/%d containing %d items - item total: %d" %
                 (saved_idx+1, self.validation_logits_save_counter, chunk_size, curr_item_idx))

        # compute partial accuracies for the unsaved chunk in item_logits
        if len(self.item_logits) > 0:
            chunk_size = len(self.item_logits)
            labels = self.item_labels[curr_item_idx:curr_item_idx + chunk_size, :]
            accuracies.append(self.get_chunk_accuracy(self.item_logits, labels))
            curr_item_idx += chunk_size
            info("Processed existing chunk containing %d items - item total: %d" % ( chunk_size, curr_item_idx))

        accuracy = np.mean(accuracies)
        return accuracy

    def get_chunk_accuracy(self, logits, labels):
        predicted_classes = np.argmax(logits, axis=1)
        correct_classes = np.argmax(labels, axis=1)
        return  np.mean(np.equal(predicted_classes, correct_classes))

