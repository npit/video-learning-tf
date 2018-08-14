from sklearn.metrics import f1_score, precision_score, recall_score
import argparse
import os
import pickle
import numpy
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("logits_path")
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()
path_logits= args.logits_path
print("Logits:", path_logits)

# deduce dataset from the local config file
path_dset = None
run_folder = os.path.dirname(path_logits)
config_file = [x for x in os.listdir(run_folder) if x.startswith("config") and x.endswith(".yml") and "graph-" not in x]
assert len(config_file) == 1, "Multiple config files deduced: %s" % config_file
config_file = os.path.join(run_folder, config_file[0])
if args.verbose:
    print("Deduced base config file:", config_file)

with open(config_file, "r") as f:
    data = yaml.load(f)["run"]["data"]
    for datname in data:
        if args.verbose:
            print("Checking data path:", datname)
        dat = data[datname]
        if "phase" not in dat:
            print("Missing phase information in dataset definitions")
            exit()
        if dat["phase"] != "defs.phase.val":
            continue
        if dat["tag"] != "defs.dataset_tag.main":
            continue
        if path_dset is not None:
            if args.verbose:
                print("Path already configured to", path_dset)
        if args.verbose:
            print("Deduced dataset identifier:", datname)
        path_dset = dat["data_path"]

if args.verbose:
    print("Deduced data path:", path_dset)

# read labels
labels=[]
logits=None
with open(path_dset) as f:
    for line in f:
        lbl = int(line.strip().split()[-1])
        labels.append(lbl)

with open(path_logits,"rb") as f:
    logits = pickle.load(f)

amax = numpy.argmax(logits, axis=1)

# print("Accuracy:")
sep=","
for avg in ["macro", "micro"]:
    print(avg, "F1:", f1_score(labels, amax.tolist(),average=avg), sep=sep)
    print(avg, "P:", precision_score(labels, amax.tolist(),average=avg), sep=sep)
    print(avg, "R:", recall_score(labels, amax.tolist(),average=avg), sep=sep)
