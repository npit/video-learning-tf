import argparse
import numpy as np
import pickle
import os

def labels_consistent(labels):
    labels1, labels2 = labels
    for l1,l2  in zip(labels1, labels2):
        if l1 != l2:
            return False
    return True

def get_accuracy(logits, labels):
    log_amax = np.argmax(logits, 1)
    return np.sum(np.equal(log_amax, labels)) / len(labels)

parser = argparse.ArgumentParser()
parser.add_argument("-paths",nargs='+', help = "Path to the input logit(s) file.")
parser.add_argument("-gt", nargs='+',help="Path to configuration or dataset file(s), one per input logits")
parser.add_argument("-lbl", nargs='+',help="Path to label indexing file", required=False)
parser.add_argument("-align", help="Align labels")
compromises=["intersect"]
parser.add_argument("-compromise", help="Fix mismatches")

args = parser.parse_args()



# read label index info
classname2idx=[]
idx2classname=[]
label_index = []

if args.lbl:
    for i, lblpath in enumerate(args.lbl):
        print(i+1,":", lblpath)
        print("Reading lbl file", lblpath)
        c2i, i2c = {}, {}
        with open(lblpath) as f:
            for line in f:
                line = line.strip()
                #print(line)
                classname, classidx = line.split()
                c2i[classname] = int(classidx)
                i2c[int(classidx)] = classname
                if int(classidx)==30:
                    print("name for 30:",classname)
        print("Read {} classes ".format(len(c2i)))
        print("Read {} classes ".format(len(i2c)))
        assert len(c2i) == len(i2c), "classnames dict lengths error"
        print(sorted(i2c.keys()))
        classname2idx.append(c2i)
        idx2classname.append(i2c)
    #    for k,v in idx2classname[-1].items():
    #        print(k, v)


# read logits
logits = []
print("Logits:")
for i, path in enumerate(args.paths):
    print(i+1,":", path)
    if os.path.basename(path).startswith("validation"):
         # load validation logits
        with open(path, 'rb') as f:
            raw_logits = pickle.load(f)

    print("Read {} logits".format(len(raw_logits)))
    logits.append(raw_logits)

# read gt
labels, vids = [], []
print("GT files:", args.gt)
for gtfile in args.gt:
    print("Reading gt file", gtfile)
    file_labels = []
    file_vids = []
    with open(gtfile) as f:
        for line in f:
            vid, label = line.strip().split()
            if vid == "XHukxF8iWE0":
                print("LABEL:",label)
            file_labels.append(int(label))
            file_vids.append(os.path.basename(vid))
    file_labels=np.asarray(file_labels)
    labels.append(file_labels)
    vids.append(file_vids)
    print("Read {} vidids and {} labels".format(len(file_vids), len(file_labels)))


labels_aligned = []
# read alignment, only 1st
if args.align:
    align_c2i = {}
    align_i2c = {}
    args.align = args.align.split()
    print(args.align)
    apath, pathidx = args.align[0], int(args.align[1])
    with open(apath) as f:
        for line in f:
            line = line.strip()
            #print(line)
            classname, classidx = line.split()
            align_c2i[classname] = int(classidx)
            align_i2c[int(classidx)] = classname
    print("Applying align to dataset 1:")
    # get classnames of unaligned dataset1 idxs
    for lbl in labels[0]:
        classname = idx2classname[0][lbl]
        newidx = align_c2i[classname]
        labels_aligned.append(newidx)
    labels[0] = labels_aligned



if len(logits) == 1:
    print(get_accuracy(logits[0],labels))
    exit(1)
assert len(logits) == 2, "Can only combine two runs."
assert len(logits[0]) == len(labels[0]), "Mismatch in logits / labels #1 lengths: %d  , %d" % ( len(logits[0]), len(labels[0]))
assert len(logits[1]) == len(labels[1]), "Mismatch in logits / labels #2 lengths: %d  , %d" % ( len(logits[1]), len(labels[1]))


print("Marginal accuracies:")
for i in range(len(args.paths)):
    accuracy = get_accuracy(logits[i], labels[i])
    print("Logits # %d/%d:" % (i+1, len(args.paths)), args.paths[i], round(accuracy,5))


if len(logits[0]) != len(logits[1]):
    print("Logit len mismatch : %s" % str(list(map(len,logits))))
    if args.compromise == "intersect":
        print("Keeping only common vid ids")
        common_vids = []
        common_labels = []
        common_logit_idxs = [[], []]
        # keep only common videos in both settings
        # preserve order of first configuration

        for i in range(len(vids[0])):
            vid = vids[0][i]
            label = labels[0][i]
            if args.align:
                classname = align_i2c[int(label)]
            else:
                classname = idx2classname[0][int(label)]
            if vid in vids[1]:
                other_idx = vids[1].index(vid)
                other_label = labels[1][other_idx]
                other_classname = idx2classname[1][int(other_label)]
                if classname != other_classname:
                    print("classnames mismatch:",classname, other_classname, "for video",vid)
                    print("Dataset1:", label, classname, i)
                    print("Dataset2:", other_label, other_classname, other_idx)
                    exit()
                common_vids.append(vid)
                common_labels.append(label)
                common_logit_idxs[0].append(i)

                common_logit_idxs[1].append(other_idx)
        logits[0], logits[1] = logits[0][common_logit_idxs[0]], logits[1][common_logit_idxs[1]]
        labels = common_labels
        vids = common_vids
    else:
        print("No compromise selected to fix mismatch, exiting.")
        exit()
else: 
    # check consistency of labels
    assert labels_consistent(labels), "Inconsistent labels."
    # label gt is identical. Pick one.
    labels = labels[0]

# dual combine
weights = [ round(x * 0.1,1) for x in range(0,11)]
weights = zip(weights, weights[-1::-1])

print("w1\tw2\tmean acc.")
for w, ww in weights:
    newlogits = w * logits[0] + ww * logits[1]
    accuracy = get_accuracy(newlogits, labels)
    print( w, ww, round(accuracy,5),sep="\t")
