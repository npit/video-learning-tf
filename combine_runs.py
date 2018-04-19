import pickle
import numpy as np
from os import makedirs, listdir, devnull
from os.path import exists, join, basename, isdir, abspath, isfile
import configparser
import argparse
import subprocess
from serialize import deserialize_from_tfrecord
import tensorflow as tf
import tqdm

"""
Script to combine validation runs
"""

def load_labels(config):
    # load the labels
    datapath = eval(config['data_path'])
    if isdir(datapath):
        datafile = join(datapath, "data.test")
    else:
        datafile = datapath + ".tfrecord"
    sizefile = datafile+".size"
    with open(sizefile, "r") as f:
        lines = []
        for line in f:
            lines.append(line)
        num = int(lines[0].strip().split()[-1])
        cpi_rlc = eval(lines[2].strip().split('\t')[-1])
        # expand RLC encoding
        cpi = [t[1] for t in cpi_rlc for _ in range(t[0])]
        fpc = eval(lines[3].split()[-1])

    labels = []
    iter = tf.python_io.tf_record_iterator(datafile)
    if not iter:
        print("Unable to initialize tfrecord iterator at file", datafile)
        exit(1)
    print("Reading %d labels from [%s]..." % (num, datafile))
    with tqdm.tqdm(total=num) as pbar:
        while True:
            try:
                _, item_labels = deserialize_from_tfrecord(iter, 1)
                if not item_labels:
                    break
                labels.extend(item_labels)
                pbar.set_description("%d labels read" % len(labels))
                pbar.update()
            except StopIteration:
                break
            except Exception as ex:
                print("Exception occured:", ex)
                exit(1)

    print("Done reading %d labels." % len(labels))
    # aggregate labels to one per item
    # we are asserting one single label per item
    # so we take the label of the first clip
    label_idx = 0
    labels_per_item = []
    for num_clips_for_item in cpi:
        labels_per_item.append(labels[label_idx])
        label_idx += num_clips_for_item * fpc
    labels = np.asarray(labels_per_item)

    print("Aggregated from cpi: %s and fpc: %s to  %d labels." % (str(cpi_rlc), str(fpc), len(labels)))
    return num, labels


def get_accuracy(logits, labels):
    return np.mean(np.equal(logits, labels))


def show_results(logits, labels):
    print("Amean of logits: %f" % get_accuracy(np.mean(logits), labels))
    if not np.any([l < 0 for l in logits]):
        gmean_logits = np.prod(logits,axis=0)**(1.0/len(logits))
        print("Gmean of logits: %f" % get_accuracy(gmean_logits, labels))
    else:
        print("[Gmean not applicable, negative logits exist]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('run_folders', nargs='+', help='Run folders or validation files to include in the aggregation.')
    parser.add_argument('config_file', help='The validation configuration file to use.')
    parser.add_argument('-mode', choices=["amean"], help='How to aggregate the partial results')
    parser.add_argument('-class_mask', help='Mask to limit the logit classes')

    args = parser.parse_args()
    print(args)
    args.run_folders = [abspath(x) for x in args.run_folders]
    args.config_file = abspath(args.config_file)

    print("Combining runs in mode: %s" % args.mode)
    print("Using config file %s" % basename(args.config_file))

    configp = configparser.ConfigParser()
    configp.read(args.config_file)
    config = configp['run']
    num_classes = int(config['num_classes'])
    num_items, labels = load_labels(config)

    logits_per_run = []
    accuracies_per_run = []
    for source_path in args.run_folders:
        if not exists(source_path):
            print(source_path, "does not exist!")
            exit(1)

        if isdir(source_path):
            # directory, get all logits chunks, in order
            val_files = sorted([f for f in listdir(source_path) if f.startswith("validation_logits") and isfile(join(source_path, f))])
            run_logits = np.zeros([0, num_classes], np.float32)
            for valfile in val_files:
                valfile = join(source_path, valfile)
                with open(valfile, "rb") as f:
                     res = pickle.load(f)
                     run_logits = np.vstack((run_logits, res))
        else:
            # file, read it
            with open(source_path,"rb") as f:
                run_logits= pickle.load(f)

        run_predictions = np.argmax(run_logits, axis=1)
        local_accuracy = np.mean(np.equal(labels, run_predictions))
        logits_per_run.append(run_logits)
        accuracies_per_run.append(local_accuracy)
        print("Aggregating run [%s] with local accuracy [%s]" % (basename(source_path), local_accuracy))

    show_results(logits_per_run, labels)
    print("Average accuracies: %f" % np.mean(accuracies_per_run))
    print("Softmax:")
    sm_logits_per_run = [np.exp(l) for l in logits_per_run]
    denom = np.sum(sm_logits_per_run)
    sm_logits_per_run = [l/denom for l in sm_logits_per_run]
    show_results(sm_logits_per_run, labels)

    # weighted average: for unequal weights, try 2x equal weight per participant
    print("Weighted:")
    w = 1/len(logits_per_run)
    print("Equal weights are:", w)
    big_w, other_w = 2*w, (1-2*w) / (len(logits_per_run)-1)
    for i in range(len(logits_per_run)):
        w_vector = [other_w for _ in logits_per_run]
        w_vector[i] = big_w
        wlogits_per_run = [l*w for (l, w) in zip(logits_per_run, w_vector)]
        print("Weights:", w_vector)
        show_results(wlogits_per_run, labels)
