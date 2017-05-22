import os
import tensorflow as tf
from scipy.misc import imread, imresize, imsave
import numpy as np
import matplotlib.pyplot as plt
import logging, time

# input test/train
input_files = [
    "/home/nik/uoa/msc-thesis/implementation/examples/test_run/frames.train",
"/home/nik/uoa/msc-thesis/implementation/examples/test_run/frames.test",
"/home/nik/uoa/msc-thesis/implementation/examples/test_run/videos.train",
"/home/nik/uoa/msc-thesis/implementation/examples/test_run/videos.test"
]

# frames or videos
frames, videos = range(2)
input_modes = [frames, frames, videos, videos]
#input_mode = "videos"

image_shape = (227,227,3)

mean_image = [103.939, 116.779, 123.68]
height = image_shape[0]
width = image_shape[1]
blue = np.full((height, width), mean_image[0])
green = np.full((height, width), mean_image[1])
red = np.full((height, width), mean_image[2])
mean_image = np.stack([blue, green, red])
mean_image = np.transpose(mean_image,[1,2,0])
mean_image = np.ndarray.astype(mean_image,np.uint8)
batchsize = 128

num_frames_per_video = 16
imageFormat = "jpg"

# datetime for timestamps
def get_datetime_str():
    #return time.strftime("[%d|%m|%y]_[%H:%M:%S]")
    return time.strftime("%d%m%y_%H%M%S")

# configure logging settings
def configure_logging():
    logging_level = logging.INFO

    logfile = "serialize_" + get_datetime_str() + ".log"
    print("Using logfile: %s" % logfile)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging_level)

    formatter = logging.Formatter('%(asctime)s| %(levelname)7s - %(filename)15s - line %(lineno)4d - %(message)s')

    # # file handler
    # handler = logging.FileHandler(logfile)
    # handler.setLevel(logging_level)
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)

    # console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(consoleHandler)
    return logger


logger = configure_logging()
# helper tfrecord function
def _int64_feature( value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# helper tfrecord function
def _bytes_feature( value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_to_tfrecord( paths, labels, outfile, mode):
    writer = tf.python_io.TFRecordWriter(outfile + ".tfrecord")

    count = 0
    for path in paths:
        if mode == videos:
            frames = get_video_frames(path)
        else:
            frames = [ read_image(path) ]
        for frame in frames:

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_shape[0]),
                'width': _int64_feature(image_shape[1]),
                'depth': _int64_feature(image_shape[2]),
                'label': _int64_feature(int(labels[count])),
                'image_raw': _bytes_feature(frame.tostring())}))
            writer.write(example.SerializeToString())
        count += 1
    writer.close()

# read all frames for a video
def get_video_frames(path):

    frames = []
    basename = os.path.basename(path)
    for im in range(num_frames_per_video):
        impath = path + ".%04d" % (1 + im) + "." + "jpg"
        frames.append(read_image(impath))
    return frames



 # read image from disk
def read_image(imagepath, useMeanCorrection=False):
    image = imread(imagepath)

    logger.debug("Reading image %s" % imagepath)
    # for grayscale images, duplicate
    # intensity to color channels
    if len(image.shape) <= 2:
        image = np.repeat(image[:, :, np.newaxis], 3, 2)
    # drop channels other than RGB
    image = image[:,:,:3]
    #  convert to BGR
    image = image[:, :, ::-1]
    # resize
    image = imresize(image, image_shape)

    # there is a problem if we want to store mean-subtracted images, as we'll have to store a float per pixel
    # => 4 x the space of a uint8 image
    # image = image - mean_image


    return image


def display_image(image,label=None):
    print(label)
    plt.title(label)
    plt.imshow(image)
    plt.show()
    # plt.waitforbuttonpress()


def deserialize_example(ex):
    features = tf.parse_single_example(
        ex,
        # Defaults are not specified since both keys are required.
        features={
            'height' : tf.FixedLenFeature([],tf.int64)
            ,'width': tf.FixedLenFeature([], tf.int64)
            ,'depth': tf.FixedLenFeature([], tf.int64)
            ,'image_raw': tf.FixedLenFeature([], tf.string)
            #, 'label': tf.FixedLenFeature([], tf.int64)
        })
    return tf.decode_raw(features['image_raw'], tf.uint8)

def deserialize_single(str_data):
    example = tf.train.Example()
    example.ParseFromString(str_data)

    height = int(example.features.feature['height']
                 .int64_list
                 .value[0])
    width = int(example.features.feature['width']
                .int64_list
                .value[0])
    img_string = (example.features.feature['image_raw']
                  .bytes_list
                  .value[0])
    depth = (example.features.feature['depth']
             .int64_list
             .value[0])
    # label = (example.features.feature['label']
    #          .int64_list
    #          .value[0])
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    # watch it : hardcoding preferd dimensions according to the dataset object.
    # it should be the shape of the stored image instead, for generic use
    image = img_1d.reshape((image_shape[0], image_shape[1], image_shape[2]))
    return image


def deserialize_from_tfrecord(self, iterator, images_per_iteration):
    # images_per_iteration :
    images = []
    for _ in range(images_per_iteration * num_frames_per_video):
        try:
            string_record = next(iterator)

            image = deserialize_single(string_record)


            subtract_mean = True
            if subtract_mean:
                image = np.ndarray.astype(image, np.float32) - mean_image
            images.append(image)
            # labels.append(label)
            # imsave('reconstructedBGR.JPEG', image)
            # image = image[:, :, ::-1] # swap 1st and 3rd dimension
            # imsave('reconstructedBGR__2.JPEG', image)
        except StopIteration:
            break
        except Exception as ex:
            logger.error('Exception at reading image, loading from scratch')
            logger.error(ex)

    # return images, labels
    return images

def write():
    for idx in range(len(input_files)):

        mode = input_modes[idx]
        inp = input_files[idx]
        logger.info("Serializing %s in mode %s" % (inp,mode))
        paths = []
        labels = []
        with open(inp,'r') as f:
            for line in f:
                path, label = line.split(' ')
                paths.append(path.strip())
                labels.append(int(label.strip()))


        serialize_to_tfrecord(paths, labels,inp, mode)
def read():
    for idx in range(len(input_files)):
        mode = input_modes[idx]
        inp = input_files[idx] + ".tfrecord"
        iter = tf.python_io.tf_record_iterator(path=inp)
        reader = tf.TFRecordReader()

        tfr = "/home/nik/uoa/msc-thesis/implementation/examples/test_run/frames.train.tfrecord"
        sss = tf.train.string_input_producer([tfr])
        l = ["/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_CricketShot_g23_c05/v_CricketShot_g23_c05.0092.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_WritingOnBoard_g10_c01/v_WritingOnBoard_g10_c01.0023.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_Rowing_g21_c05/v_Rowing_g21_c05.0112.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_WallPushups_g13_c02/v_WallPushups_g13_c02.0095.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_Diving_g21_c03/v_Diving_g21_c03.0034.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_JavelinThrow_g13_c03/v_JavelinThrow_g13_c03.0061.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_PlayingSitar_g09_c02/v_PlayingSitar_g09_c02.0040.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_HorseRiding_g20_c06/v_HorseRiding_g20_c06.0078.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_PlayingDaf_g20_c01/v_PlayingDaf_g20_c01.0053.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_WallPushups_g14_c02/v_WallPushups_g14_c02.0111.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_MilitaryParade_g09_c04/v_MilitaryParade_g09_c04.0066.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_PullUps_g18_c01/v_PullUps_g18_c01.0036.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_HorseRace_g19_c05/v_HorseRace_g19_c05.0290.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_BasketballDunk_g21_c03/v_BasketballDunk_g21_c03.0015.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_PlayingDaf_g22_c03/v_PlayingDaf_g22_c03.0048.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_HulaHoop_g17_c01/v_HulaHoop_g17_c01.0019.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_Rowing_g10_c05/v_Rowing_g10_c05.0012.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_HulaHoop_g13_c01/v_HulaHoop_g13_c01.0059.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_BoxingSpeedBag_g15_c04/v_BoxingSpeedBag_g15_c04.0171.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_HighJump_g23_c01/v_HighJump_g23_c01.0052.jpg", "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/v_GolfSwing_g20_c07/v_GolfSwing_g20_c07.0087.jpg"]
        filename_queue = tf.train.string_input_producer(l)
        _, example = reader.read(filename_queue)
        im = deserialize_example(example)

        s = tf.Session()
        s.run(tf.global_variables_initializer())
        aa = s.run(sss)

        print(s.run(im))
        next(iter)

        # use given converter apis?
# https://kwotsin.github.io/tech/2017/01/29/tfrecords.html


# read()
write()