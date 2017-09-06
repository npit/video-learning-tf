import sys
from math import ceil
'''
Script to project the learning rate drop wrt to other training parameters
'''

def print_projections(decayrange, total_batches, baselr, decay):
    print("Learning rate projections:")
    for n, num_values in enumerate(decayrange):
        interval = round(total_batches / num_values)
        last_lr = baselr * pow(decay, num_values)
        print("Num. lr values=%5d: lr=%2.6f decay interval=%5d" % (num_values, last_lr, interval))

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: %s numdata numepochs batchsize baselr decay=0.96" % sys.argv[0])
        exit(1)
    
    numdata = int(sys.argv[1])
    epochs = int(sys.argv[2])
    batchsize = int(sys.argv[3])
    baselr = float(sys.argv[4])
    if len(sys.argv) > 5:
        decay = float(sys.argv[5])
    else:
        decay = 0.96

    total_num_values = [5, 10, 25, 50, 100, 200, 500]

    batches_per_epoch = ceil(numdata / batchsize)
    total_batches = batches_per_epoch * epochs
    print("Number of %d-sized batches per epoch of %d data: %d" % (batchsize, numdata, batches_per_epoch))
    print("Number of batches for all %d epochs: %d" % (epochs, total_batches))

    print_projections(total_num_values, total_batches, baselr, decay)
    while True:
        res = input("Enter value(s) for number of learning rate values to check: ")
        try:
            total_num_values = list(map(int,res.split()))
        except Exception:
            print("Gotta be numbers.")
        print_projections(total_num_values, total_batches, baselr, decay)



