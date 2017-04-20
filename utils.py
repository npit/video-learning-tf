import numpy as np
# error function
def error(msg):

    raise Exception(msg)

def labels_to_one_hot(labels,num_classes):
    onehots = np.zeros(shape=(len(labels),num_classes),dtype=np.int32)

    for l in range(len(labels)):
        onehots[l][labels[l]] = 1
    return onehots