import argparse
import numpy as np
import pickle
import os

def get_accuracy(logits, labels):
    log_amax = np.argmax(logits, 1)
    return np.sum(np.equal(log_amax, labels)) / len(labels)

parser = argparse.ArgumentParser()
parser.add_argument("-paths",nargs='+')
parser.add_argument("-gt",help="Path to configuration file or dataset file")
parser.add_argument("-aidx_correction", help="File with indexes to adjust audio logit order and inclusion")
parser.add_argument("-flbl_correction", help="File with frame to audio label mappings")

args = parser.parse_args()



aidx = []
if args.aidx_correction:
    with open(args.aidx_correction) as ff:
        for line in ff:
            aidx.append(int(line.strip()))
fidx = []
if args.flbl_correction:
    with open(args.flbl_correction) as ff:
        for line in ff:
            fidx.append(int(line.strip().split()[-1]))

# read logits
logits = []
print("Logits:")
for i, path in enumerate(args.paths):
    print(i+1,":", path)
    if os.path.basename(path).startswith("validation"):
         # load validation logits
        with open(path, 'rb') as f:
            raw_logits = pickle.load(f)

    if i == 1 and args.aidx_correction:
        print("Reordering audio logits:", path)
        # reorder audio logits
        raw_logits = raw_logits[np.asarray(aidx)]
    if i == 0 and fidx:
        print("Reordering frame labels")
        raw_logits = raw_logits[:, np.asarray(fidx)]
    logits.append(raw_logits)

# read gt
labels, labelids = [], []
print("Labels file:", args.gt)
with open(args.gt) as f:
    for line in f:
        vid, label = line.strip().split()
        labels.append(int(label))
        labelids.append(os.path.basename(vid))
    labels=np.asarray(labels)
    if args.aidx_correction:
        # reorder 
        print("Reordering gt labels. First 10 un-reordered:", labels[:10])
        idx = []
        with open(args.aidx_correction) as ff:
            for line in ff:
                idx.append(int(line.strip()))
        labels = labels[idx]
        labelids=[labelids[i] for i in idx]
    print("First 10 labels:")
    print(labels[:10])
    print("First 10 ids:")
    print(labelids[:10])

if len(logits) == 1:
    print(get_accuracy(logits[0],labels))
    exit(1)
assert len(logits) == 2, "Can only combine two runs."
assert len(logits[0]) == len(logits[1]), "Logit len mismatch : %s" % str(list(map(len,logits)))
assert len(logits[0]) == len(labels), "Mismatch in logits / labels lengths: %d  , %d" % ( len(logits[0]), len(labels))
# dual combine
weights = [ round(x * 0.1,1) for x in range(0,11)]
weights = zip(weights, weights[-1::-1])

print("w1\tw2\tmean acc.")
for w, ww in weights:
    newlogits = w * logits[0] + ww * logits[1]
    accuracy = get_accuracy(newlogits, labels)
    print( w, ww, round(accuracy,5),sep="\t")
