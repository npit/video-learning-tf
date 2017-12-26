import os
import sys
import tensorflow as tf
import numpy as np
from collections import OrderedDict
import tqdm

# read from tfrecord
def deserialize_from_tfrecord(iterator):
    # images_per_iteration :
    images = []
    labels  = []
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

    except StopIteration as ex:
        #print("Encountered stop iteration exception.")
        return None,"stop" + str(ex),None

    except Exception as ex:
        #print('Exception at reading image')
        return None,"Exception: " + str(ex),None

    images.append(image)
    labels.append(label)

    return images, labels, len(string_record)

def read_size_file(sizefile):
    if os.path.exists(sizefile):
        print("Size file contents:")
        print("------------------")
        lines = []
        with open(sizefile,"r") as f:
            for line in f:
                line = line.strip()
                print(line)
                lines.append(" ".join(line.split()[1:]))
        cpv = eval(lines[2])[0]
        fpc = int(lines[3])
        numframes = sum([fpc * cpv[1] for _ in range(cpv[0])])
        print("------------------")
    else:
        print("File %s does not exist." % sizefile)
        exit(1)
    return numframes

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Give an input .tfrecord file")
        exit(1)
    if len(sys.argv) > 2:
        opts = sys.argv[2:]
    else:
        opts = []
    num_items = None
    if opts:
        try:
            num_items = int(opts[0])
        except Exception as ex:
            print(ex)
            exit(1)

    filename = sys.argv[1]
    print("Examining contents of %s" % filename)
    sizefile = filename + ".size"
    expected_num = read_size_file(sizefile)

    if num_items is not None:
        print("Printing %s items due to user argument" % str(num_items))

    iterator = tf.python_io.tf_record_iterator(path = filename)
    shapes = OrderedDict()
    count = 0
    strlens = {}
    messages = []
    with tqdm.tqdm() as pbar:
        while(True):
            try:
                images, ex, strlen = deserialize_from_tfrecord(iterator)
                if not images:
                    if ex == "stop":
                        if count != expected_num:
                            messages.append("Read %d items, but the sizefile says %d" % (count, expected_num))
                    break
                count += 1
                shp = images[-1].shape
                if not shp in shapes:
                    print("\nNew shape encountered:", shp)
                    shapes[shp] = 0
                shapes[shp] += 1
                if not strlen in strlens:
                    strlens[strlen] = 0
                strlens[strlen] += 1

                pbar.set_description(desc = "Processed [%d] items. " % count)
                pbar.update()
                if num_items is not None:
                    if num_items == count:
                        break
            except EOFError as ex:
                print(ex)
                break
            except StopIteration as ex:
                print(ex)
                break

    print("Item shape distribution in the file:")
    for shp in shapes:
        print(shp,shapes[shp])
    print("String record length distribution in the file:")
    for s in strlens:
        print(s, strlens[s])

    if not messages:
        print("Data is OK.")
    else:
        for msg in messages:
            print(msg)


