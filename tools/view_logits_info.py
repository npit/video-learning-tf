from itertools import groupby
import sys
import pickle
import numpy as np


def main(argv):
    if not argv:
        print("Need path(s) to logits.")
        exit(1)

    logitsfiles = argv[:]
    for i, lfile in enumerate(logitsfiles):
        with open(lfile, "rb") as f:
            logits = pickle.load(f)
        print("File: %d/%d [%s]" % (i+1, len(logitsfiles), lfile))
        print("\tshp: ", logits.shape)
        amax1 = np.argmax(logits, axis=1)
        # to rlc
        amax1 = [list(g) for k,g in groupby(amax1)]
        amax1 = [(k[0],len(k)) for k in amax1]

        print("\targmax-1: ", amax1)
        nclasses = int(logits.shape[-1])
        hist = []
        print("\tOccurence for each of the %d classes:" % nclasses)
        for n in range(nclasses):
            s = sum([x[1] if x[0] == n else 0 for x in amax1])
            print("\t\tclass %4d: %4f %% " % (n, s/len(logits)*100.0))
            hist.append((s, n))
        hist = sorted(hist, key=lambda x: x[0])

        print("\tClasses sorted by occurence:")
        for val, name in hist:
            print("\t\t%d %f %%" % (name, val / len(logits) * 100.0))


if __name__ == "__main__":
    main(sys.argv[1:])


