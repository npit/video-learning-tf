import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import pickle
import argparse
import tqdm
from serialize import deserialize_from_tfrecord

"""
Produces a mean image from a collection of serialized tfrecord files
"""
def main():
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path",help="The path to the file list & labels")
    args = parser.parse_args()

    # read number of data
    tfrecord_file = args.file_path + ".tfrecord"
    size_file = tfrecord_file + ".size"
    with open(size_file) as f:
        for line in f:
            num_items = int(line.split()[-1])
            break
    # compute the mean
    print("Computing the mean of %d images from %s" % (num_items, tfrecord_file))
    iter = tf.python_io.tf_record_iterator(tfrecord_file)
    mean_image = 0
    for _ in tqdm.trange(num_items):
        images, _ = deserialize_from_tfrecord(iter, 1)
        mean_image += np.float32(images[0])

    mean_image = np.uint8(mean_image / num_items)
    outpath = tfrecord_file + ".mean"
    print("\nWriting mean image to", outpath)
    # pickle image
    with open(outpath, "wb") as f:
        pickle.dump(mean_image,f)
    # imagey image
    imsave(outpath + ".png", mean_image)
    # one value per channel
    with open(outpath + ".3","w") as f:
        f.write("[")
        for i in range(3):
            f.write("%f " % np.mean(mean_image[:,:,i]))
            if i < 2:
                f.write(",")
        f.write("]")




if __name__ == '__main__':
    main()
