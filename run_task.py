# generic IO
import argparse
import lrcn_
from settings_ import *
from train import Train
from val import Validation
# project modules
from models.model import Model

# utils
from utils_ import *
from defs_ import defs


# print information on current iteration
def print_iter_info(settings, feeder, num_images, num_labels, padding):
    dataset = feeder.datasets[settings.phase][0]
    padinfo = "(%d padding)" % padding if padding > 0 else ""
    epoch_str = "" if settings.val else "epoch: %2d/%2d," % (settings.train.epoch_index+1, settings.train.epochs)
    msg = "Mode: [%s], %s batch %4d / %4d : %s images%s, %3d labels" % \
          (settings.phase, epoch_str , dataset.batch_index, len(dataset.batches), str(num_images), padinfo, num_labels)
    info(msg)


# train the network
def do_train(settings, train, feeder, model, sess, tboard_writer, summaries):

    # mop up all summaries. Unless you separate val and train, the training subgraph
    # will be executed when calling the summaries op. Which is bad.
    ops = [summaries.train_merged, train.loss, train.current_lr, train.global_step, train.optimizer]
    required_input = [i for component in [train, model] for i in component.required_input]

    run_batch_count = 0
    min_train_loss = (1000, -1)
    info("Starting train")
    for _ in range(settings.train.epoch_index, settings.train.epochs):
        while feeder.loop():
            # read  batch
            #images, ground_truth, dataset_ids = feeder.get_next_batch()
            fdict, num_data, num_labels, padding = feeder.get_feed_dict(required_input)
            print_iter_info(settings, feeder, num_data, num_labels, padding)

            # count batch iterations
            run_batch_count += 1
            summaries_train, batch_loss, learning_rate, settings.global_step, optimizer = sess.run(ops,feed_dict=fdict)
            # mark minimum training loss
            if min_train_loss[0] > batch_loss:
                min_train_loss = (batch_loss, settings.global_step)
            # calcluate the number of bits
            nats = batch_loss / math.log(settings.network.num_classes)
            info("Learning rate %2.8f, global step: %d, batch loss/nats : %2.5f / %2.3f " % \
                 (learning_rate, settings.global_step, batch_loss, nats))
            info("Dataset global step %d, epoch index %d, batch sizes %s, batch index train %d" %
                                 (settings.global_step, settings.train.epoch_index + 1, str(feeder.get_batch_sizes()),
                                  feeder.get_batch_index()))

            tboard_writer.add_summary(summaries_train, global_step=settings.global_step)
            tboard_writer.flush()

            # check if we need to save
            if feeder.should_save(run_batch_count):
                # save a checkpoint if needed
                progress_str = "ep_%d_btch_%d_gs_%d" %  (1 + settings.train.epoch_index, feeder.get_batch_index(), settings.global_step)
                feeder.save(sess, progress_str,  settings.global_step)
        # print finished epoch information
        if run_batch_count > 0:
            info("Epoch [%d] training run complete." % (1+settings.train.epoch_index))
        else:
            info("Resumed epoch [%d] is already complete." % (1+settings.train.epoch_index))

        settings.train.epoch_index = settings.train.epoch_index + 1
        # reset phase
        feeder.rewind_datasets()

    # Report the minimum training loss
    info("Minimum training loss: %2.2f on global index %d" % (min_train_loss[0], min_train_loss[1]))

    # if we did not save already at the just completed batch, do it now at the end of training
    if run_batch_count > 0 and not feeder.should_save(run_batch_count):
        info("Saving model checkpoint out of turn, since training's finished.")
        progress_str = "ep_%d_btch_%d_gs_%d" % (1 + settings.train.epoch_index, settings.get_num_batches(), settings.global_step)
        feeder.save(sess, progress_str, settings.global_step)


def do_test(settings, val, feeder, model, sess, tboard_writer, summaries):
    tic = time.time()

    settings.global_step = 0
    required_input = [ i for component in [val, model] for i in component.required_input ]

    # validation
    while feeder.loop():
        # get images and labels
        fdict, num_data, num_labels, padding = feeder.get_feed_dict(required_input)
        print_iter_info(settings, feeder, num_data, num_labels, padding)
        logits = sess.run(model.logits, feed_dict=fdict)
        val.process_validation_logits( defs.dataset_tag.main, settings, logits, fdict, padding)
        val.save_validation_logits_chunk()
    # save the complete output logits
    val.save_validation_logits_chunk(save_all = True)

    # done, get accuracy
    if defs.workflows.is_description(settings.workflow):
        val.process_description(settings)
    else:
        accuracy = val.get_accuracy()
        # no use in adding a single scalar accuracy summary to tensorboard
        # summaries.val.append(tf.summary.scalar('accuracyVal', accuracy))
        # summaries.merge()
        # tboard_writer.add_summary(summaries.val_merged, global_step= settings.global_step)
        info("Validation run complete in [%s], accuracy: %2.5f" % (elapsed_str(tic), accuracy))
        # if specified to save the logits, save the accuracy as well
        if val.validation_logits_save_interval is not None:
            with open(os.path.join(settings.run_folder, "accuracy_" + settings.run_id), "w") as f:
                f.write(str(accuracy))

    tboard_writer.flush()
    return True


def main(init_file):
    # create and initialize settings and dataset objects
    settings = Settings()
    feeder = settings.initialize(init_file)

    # init summaries for printage
    summaries = Summaries()

    # create and configure the model
    model = Model(settings)
    train = Train(settings, feeder, model.get_output(), summaries)
    val = Validation(settings, model.get_output())


    # create and init. session and visualization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # restore graph variables,
    feeder.init_saveload(sess, settings.resume_file, model.get_ignorable_variable_names())

    # mop up all summaries. Unless you separate val and train, the training subgraph
    # will be executed when calling the summaries op. Which is bad.
    summaries.merge()

    # create the writer for visualizashuns
    tboard_writer = tf.summary.FileWriter(settings.tensorboard_folder, sess.graph)
    if settings.train:
        do_train(settings, train, feeder, model, sess, tboard_writer, summaries)
    elif settings.val:
        do_test(settings,    val, feeder, model, sess, tboard_writer, summaries)

    # mop up
    tboard_writer.close()
    sess.close()
    info("Run [%s] complete." % settings.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("init_file", help="Configuration .ini file for the run.")
    args = parser.parse_args()

    main(args.init_file)
