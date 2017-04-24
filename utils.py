import numpy as np
import time

# datetime for timestamps
def get_datetime_str():
    return time.strftime("%d.%m.%y_%H:%M:%S")
# error function
def error(msg):

    raise Exception(msg)

def labels_to_one_hot(labels,num_classes):
    onehots = np.zeros(shape=(len(labels),num_classes),dtype=np.int32)

    for l in range(len(labels)):
        onehots[l][labels[l]] = 1
    return onehots

def print2(msg, indent=0, type="", verbose = True):
    if not verbose:
        return
    ind = ''
    for i in range(indent):
        ind+= '\t'
    bann = ""
    if type == "banner":
        for _ in msg:
            bann+='#'
        print(ind,bann)
    print (ind+msg)
    if type == "banner":
        print(ind,bann)
