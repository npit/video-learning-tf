import os
import sys

"""
Script to check the distribution of labels in a dataset
"""
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Filename needed.")
        exit(1)

    filename = sys.argv[-1]
    if not os.path.exists(filename):
        print("Filename does not exist.")
        exit(1)

    hist= {}
    length_hist = {}
    min_label = ("", sys.maxsize)
    with open(filename,"r") as f:
        for line in f:
            line = line.strip().split()
            path, labels = line[0], line[1:]
            labels = tuple(labels)
            if not labels in hist:
                hist[labels] = 0
            hist[labels] += 1
            if not len(labels) in length_hist:
                length_hist[len(labels)] = 0
            length_hist[len(labels)] += 1

    # print frequencies of each label
    print("Min label samples:")
    print("{}".format(min(list(hist.items()), key = lambda x : x[1])))
    print("Max label samples:")
    mx = max(list(hist.items()), key = lambda x : x[1])
    print("{}".format(mx))
    print("Accuracy of picking most frequent class:")
    print("%2.3f %%" % (mx[1] / (sum(hist.values())) * 100 ))

    print("Samples per label:")
    for i,label in enumerate(hist):
        print("%d/%d : %s | %d" % (i+1, len(hist), " ".join(label), hist[label]))

    # print length of each label, for vector labels
    print("\nSamples per label length:")
    for llen in length_hist:
        print("%d | %d" % (llen, length_hist[llen]))




