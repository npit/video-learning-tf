import os
import sys
import tensorflow as tf
import numpy as np
from collections import OrderedDict
import tqdm

# read from tfrecord
def deserialize_from_tfrecord(iterator, images_per_iteration):
    # images_per_iteration :
    images = []
    labels  = []
    for imidx in range(images_per_iteration):
        try:
            string_record = next(iterator)
            example = tf.train.Example()
            example.ParseFromString(string_record)
            img_string = (example.features.feature['image_raw']
                          .bytes_list
                          .value[0])
            height = int(example.features.feature['height']
                         .int64_list
                         .value[0])
            width = int(example.features.feature['width']
                        .int64_list
                        .value[0])
            depth = (example.features.feature['depth']
                     .int64_list
                     .value[0])
            label = (example.features.feature['label']
                     .int64_list
                     .value)
            label = list(label)
            label = label[0] if len(label) == 0 else label

            img_1d = np.fromstring(img_string, dtype=np.uint8)
            image = img_1d.reshape(height, width, depth)

        except StopIteration:
            #print("Stop iteration thrown.")
            return None, None

        except Exception as ex:
            #print('Exception at reading image, loading from scratch')
            return None, None

        images.append(image)
        labels.append(label)

    return images, labels


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Give an input .tfrecord file")
        exit(1)

    filename = sys.argv[-1]
    print("Examining contents of %s" % filename)
    sizefile = filename + ".size"
    if os.path.exists(sizefile):
        print("Size file contents:")
        print("------------------")
        with open(sizefile,"r") as f:
            for line in f:
                print(line.strip())
            print()

        print("------------------")

    if not os.path.exists(sizefile):
        print("File %s does not exist." % filename)
        exit(1)
    iterator = tf.python_io.tf_record_iterator(path = filename)
    shapes = OrderedDict()
    count = 0
    with tqdm.tqdm() as pbar:
        while(True):
            try:
                images, _ = deserialize_from_tfrecord(iterator,1)
                if not images:
                    break
                count += 1
                shp = images[-1].shape
                if not shp in shapes:
                    print("\nNew shape encountered:", shp)
                    shapes[shp] = 0
                shapes[shp] += 1
                pbar.set_description(desc = "Processed [%d] items. " % count)
                pbar.update()
            except EOFError as ex:
                print(ex)
                break
            except StopIteration as ex:
                print(ex)
                break

    print("Item shape distribution in the file:")
    for shp in shapes:
        print(shp,shapes[shp])


