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

args = parser.parse_args()

# read logits
logits = []
print("Logits:")
for i, path in enumerate(args.paths):
    print(i+1,":", path)
    if os.path.basename(path).startswith("validation"):
         # load validation logits
        with open(path, 'rb') as f:
            logits.append(pickle.load(f))
# read gt
labels = []
with open(args.gt) as f:
    for line in f:
        labels.append(int(line.strip().split()[-1]))

assert len(logits) == 2, "Can only combine two runs."
assert len(logits[0]) == len(logits[1]), "Logit len mismatch : %s" % str(map(len,logits))
assert len(logits[0]) == len(labels), "Mismatch in logits / labels lengths: %d  , %d" % ( len(logits), len(labels))
# dual combine
weights = [ round(x * 0.1,1) for x in range(1,10)]
weights = zip(weights, weights[-1::-1])

print("w1\tw2\tmean acc.")
for w, ww in weights:
    newlogits = w * logits[0] + ww * logits[1]
    accuracy = get_accuracy(newlogits, labels)
    print( w, ww, round(accuracy,5),sep="\t")

