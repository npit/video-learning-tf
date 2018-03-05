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


class train:
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
        log_message += ", mid / last lr is: %1.5f, %1.5f, total drops: %d" % (lr_per_batch[len(lr_per_batch)//2], lr_per_batch[-1], len(set(lr_per_batch)))
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


