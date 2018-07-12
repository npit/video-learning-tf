import argparse
import numpy as np
import pickle
import os
import pandas as pd

def labels_consistent(labels):
    labels1, labels2 = labels
    for i,(l1,l2)  in enumerate(zip(labels1, labels2)):
        if l1 != l2:
            print("Diff lbls {}, {} at index {}".format(l1, l2, i))
            return False
    return True



def get_accuracy_amax(videos, i2c, amaxes, labels):
    log_amax = np.argmax(logits, 1)
    for v,amax,lbl in zip(videos, log_amax, labels):
        cname = i2c[lbl]
        mtch = "OK" if lbl == amax else ""
        #print("{} {} {} {} {}".format(v, lbl, cname, amax, mtch))
    print(labels)
    print(log_amax)
    return np.sum(np.equal(log_amax, labels)) / len(labels)


def get_accuracy(videos, i2c, logits, labels):
    log_amax = np.argmax(logits, 1)
    for v,amax,lbl in zip(videos, log_amax, labels):
        cname = i2c[lbl]
        mtch = "OK" if lbl == amax else ""
        #print("{} {} {} {} {}".format(v, lbl, cname, amax, mtch))
    #print(labels)
    #print(log_amax)
    return np.sum(np.equal(log_amax, labels)) / len(labels)


def get_accuracy_simple(logits, labels):
    log_amax = np.argmax(logits, 1)
    return np.sum(np.equal(log_amax, labels)) / len(labels)

parser = argparse.ArgumentParser()
parser.add_argument("-paths",nargs='+', help = "Path to the input logit(s) file.")
parser.add_argument("-gt", nargs='+',help="Path to configuration or dataset file(s), one per input logits")
parser.add_argument("-lbl", nargs='+',help="Path to label indexing file", required=False)
parser.add_argument("-align", help="Align label indexes to the class names - indexes of this file")
parser.add_argument("-align_target", help="Which dataset to align")
parser.add_argument("-limit",  help="limit instances", required=False, type=int)

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
        print("Read {} and {} class <-> index mappings".format(len(c2i), len(i2c)))
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



if args.limit:
    print(args.limit)
    vids = [x[:args.limit] for x in vids]
    labels = [x[:args.limit] for x in labels]
    logits = [x[:args.limit,:] for x in logits]
    with open("logits.pickle","wb") as f:
        pickle.dump(logits, f)


# read alignment, only 1st
if args.align:
    print("Pre-align accuracies:")
    for i in range(len(args.paths)):
        if i == 1:continue
        accuracy = get_accuracy(vids[i], idx2classname[i], logits[i], labels[i])
        print("Logits # %d/%d:" % (i+1, len(args.paths)), args.paths[i],"\nAccuracy:\t", round(accuracy,5))


    print("Aligning {} logit matrices, with shapes {}".format(len(logits), [x.shape for x in logits]))

    print("Aligning via:", args.align)

    # print video - classname association prior
    for d in range(len(vids)):
        if d == 1: continue
        with open("dset_%d_to_align_prior.txt" % d, "w") as f:
            for vid, label, logit in zip(vids[d], labels[d], logits[d]):
                classname = idx2classname[d][label]
                amax = np.argmax(logit)
                f.write("{} {} {} {}\n".format(vid, label, classname, amax))

    align_c2i = {}
    align_i2c = {}
    with open(args.align) as f:
        for line in f:
            line = line.strip()
            #print(line)
            classname, classidx = line.split()
            align_c2i[classname] = int(classidx)
            align_i2c[int(classidx)] = classname


    # update logits
    logits_aligned = []

    # fix class indices everywhere
    dataset_label_pos_mapping = []
    labels_aligned = []
    for dsetIdx in range(len(labels)):
        dataset_label_pos_mapping.append({})
        labels_aligned.append([])
        for itemIdx in range(len(labels[dsetIdx])):
            itemLabel = labels[dsetIdx][itemIdx]
            # get classname of target   
            classname = idx2classname[dsetIdx][itemLabel]
            # get updated class index wrt alignment mapping
            newidx = align_c2i[classname]
            labels_aligned[-1].append(newidx)


        # make a mapping from every index to every index
        dataset_i2c = idx2classname[dsetIdx]
        for l in range(len(dataset_i2c)):
            newidx = align_c2i[dataset_i2c[l]]
            dataset_label_pos_mapping[-1][l]=newidx
        #[print(x, "-" ,align_i_old2new[x]) for x in align_i_old2new]
        mapping_vector = [dataset_label_pos_mapping[-1][x] for x in list(range(logits[dsetIdx].shape[1]))]
        # "mirror" mapping vector
        composite = sorted(zip(mapping_vector, list(range(len(mapping_vector)))), key = lambda x : x[0])
        mapping_vector = [x[1] for x in composite]
        

        print(mapping_vector)

        newlogits = logits[dsetIdx]
        print("Dataset #",dsetIdx,":")
        changedLabels = True
        if mapping_vector == list(range(len(mapping_vector))):
            print("\tmapping agrees with the aligned class-index file".format(dsetIdx))
        else:
            print("\tmapping does reordering:")
            z = zip(mapping_vector, list(range(len(mapping_vector))))
            zdiff = [x for x in z if x[0]!=x[1]]
            print(zdiff)
#        if np.array_equal(logits[dsetIdx], logits[dsetIdx][:, mapping_vector]):
#            print("\tlogit-wise agrees with the aligned class-index file".format(dsetIdx))
#        else:
#            print("\tlogits are changed")
#        if all(labels[dsetIdx] == labels_aligned[-1]):
#            print("\tlogit-wise agrees with the aligned label-wise".format(dsetIdx))
#        else:
#            print("\tlabels are changed")
        if changedLabels == True:
            newlogits = newlogits[:,mapping_vector]
        logits_aligned.append(newlogits)
        print("aligned logits now at length", len(logits_aligned))
        print("aligned labels now at length", len(labels_aligned))


    logits = logits_aligned
    labels = labels_aligned

    # print video - classname association prior
    for d in range(len(vids)):
        if d == 1: continue
        with open("dset_%d_to_align_post.txt" % d, "w") as f:
            for vid, label, logit in zip(vids[d], labels[d], logits[d]):
                classname = align_i2c[label]
                amax = np.argmax(logit)
                f.write("{} {} {} {}\n".format(vid, label, classname, amax))

#if len(logits) == 1:
#    print(get_accuracy(logits[0],labels))
#    exit(1)
assert len(logits) == 2, "Can only combine two runs."
assert len(logits[0]) == len(labels[0]), "Mismatch in logits / labels #1 lengths: %d  , %d" % ( len(logits[0]), len(labels[0]))
assert len(logits[1]) == len(labels[1]), "Mismatch in logits / labels #2 lengths: %d  , %d" % ( len(logits[1]), len(labels[1]))


print("Post-alignment accuracies:")
for i in range(len(args.paths)):
    if i == 1:continue
    accuracy = get_accuracy(vids[i], align_i2c, logits[i], labels[i])
    print("Logits # %d/%d:" % (i+1, len(args.paths)), args.paths[i],"\nAccuracy:\t", round(accuracy,5))


if len(logits[0]) != len(logits[1]):
    print("Logit len mismatch : %s" % str(list(map(len,logits))))
    if args.compromise == "intersect":
        print("Keeping only common vid ids")
        common_labels = []
        common_logit_idxs = [[], []]
        # keep only common videos in both settings
        common_vids = [v for v in vids[0] if v in vids[1]]

        # preserve order of first configuration

        for i in range(len(common_vids)):
            vid = common_vids[i]
            v_idx0 = vids[0].index(vid)
            v_idx1 = vids[1].index(vid)

            # make sure labels are common as well
            label0 = labels[0][v_idx0]
            label1 = labels[1][v_idx1]

            name0 = align_i2c[int(label0)]
            name1 = align_i2c[int(label1)]

            if label0 != label1:
                print("Video {}, idxs {}, {}, label idxs {}, {} has different labels {}, {}".format(vid, v_idx0, v_idx1, label0, label1, name0, name1))
                print("idx/label/classname: {} {} {}".format(v_idx0,  label0,  name0, ))
                print("idx/label/classname: {} {} {}".format( v_idx1,  label1,  name1))
                exit(1)
            if name0 != name1:
                print("Video {}, idxs {}, {}, label idxs {}, {} has different names {}, {}".format(vid, v_idx0, v_idx1, label0, label1, name0, name1))
                print("idx/label/classname: {} {} {}".format(v_idx0,  label0,  name0, ))
                print("idx/label/classname: {} {} {}".format( v_idx1,  label1,  name1))
                exit(1)
            common_logit_idxs[0].append(v_idx0)
            common_logit_idxs[1].append(v_idx1)
            common_labels.append(label0)
        logits=[logits[0][common_logit_idxs[0]], logits[1][common_logit_idxs[1]]]
        labels = common_labels


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
    accuracy = get_accuracy_simple(newlogits, labels)
    print( w, ww, round(accuracy,5),sep="\t")
