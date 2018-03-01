import os
import sys
import tensorflow as tf
import numpy as np
from collections import OrderedDict
import tqdm
import argparse

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
        numitems = int(lines[0])
        cpv = eval(lines[2])[0]
        fpc = int(lines[3])
        numframes = sum([fpc * cpv[1] for _ in range(cpv[0])])
        print("------------------")
    else:
        print("File %s does not exist." % sizefile)
        exit(1)
    return numframes, numitems, str(cpv), fpc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tfrecord_path")
    parser.add_argument("--print_max",nargs=1,type=int)
    parser.add_argument("--verbose",action="store_true")
    args = parser.parse_args()

    filename = args.tfrecord_path
    print("Examining contents of %s" % filename)
    sizefile = filename + ".size"
    expected_num, numitems, cpv, fpc = read_size_file(sizefile)

    if args.print_max:
        print("Printing %s items due to user argument" % str(args.print_max))

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
                    if args.verbose:
                        print("No images retrieved, count:",count)
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
            except EOFError as ex:
                print(ex)
                break
            except StopIteration as ex:
                print(ex)
                break

    print("Item shape distribution in the file:")
    for shp in shapes:
        print("shape:",shp,"#items with that shape:",shapes[shp])
    print("String record length distribution in the file:")
    for s in strlens:
        print("strlen:",s,"#items with that len:", strlens[s])

    if not messages:
        print("Data is OK: count: %d, sizefile expected: %d = %d x %s x %d" % (count, expected_num, numitems, cpv, fpc))
    else:
        for msg in messages:
            print(msg)


