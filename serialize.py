import tensorflow as tf
import numpy as np
from random import shuffle
from scipy.misc import imread, imresize, imsave

import logging, time, threading, os
from utils_ import *
import configparser
import sys

'''
Script for production of training / testing data collections and serialization to tf.record files.
'''
class serialization_settings:
    init_file = "config.ini"

    # necessary config. variables
    input_files = []

    # defaults
    path_prepend_folder = None
    num_threads = 4
    num_items_per_thread = 500
    num_frames_per_clip = 16
    raw_image_shape = (240,320,3)
    clipframe_mode = defs.clipframe_mode.rand_clips
    clip_offset_or_num = 1
    frame_format = "jpg"
    force_video_metadata = False
    do_shuffle = False
    do_serialize = False
    do_validate = True
    validate_pcnt = 10
    # internals
    max_num_labels = -1
    logger = None
    logging_level = logging.INFO

    # initialize from file
    def initialize_from_file(self,argv):
        if len(argv) > 1:
            self.init_file = argv[-1]
        if self.init_file is None:
            return
        if not os.path.exists(self.init_file):
            return
        tag_to_read = "serialize"
        print("Initializing from file %s" % self.init_file)
        config = configparser.ConfigParser()
        config.read(self.init_file)
        if not config[tag_to_read ]:
            error('Expected header [%s] in the configuration file!' % tag_to_read)
        config = config[tag_to_read]
        for key in config:
            exec("self.%s=%s" % (key, config[key]))
        # initialize logging
        self.logger = CustomLogger()
        logfile = "log_serialize_" + get_datetime_str() + ".log"
        self.logger.configure_logging(logfile, self.logging_level)
        print("Successfully initialized from file %s" % self.init_file)



# datetime for timestamps
def get_datetime_str():
    #return time.strftime("[%d|%m|%y]_[%H:%M:%S]")
    return time.strftime("%d%m%y_%H%M%S")


# helper tfrecord function
def _int64_feature( value):
    if not type(value) == list:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# helper tfrecord function
def _bytes_feature( value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# write data metadata
def write_size_file(item_paths, clips_per_item, outfile, mode, max_num_labels, settings):
    if mode == defs.input_mode.image:
        settings.num_frames_per_clip = None
    with open(outfile + ".size", "w") as f:
        # do the write
        f.write("items\t%d\n" % len(item_paths))
        f.write("type\t%s\n" % mode)
        f.write("cpi\t%s\n" % str(clips_per_item))
        f.write("fpc\t%s\n" % str(settings.num_frames_per_clip))
        f.write("labelcount\t%s\n" % str(max_num_labels))




def serialize_multithread(item_paths, clips_per_item, frame_paths, labels, outfile, mode, max_num_labels, settings):

    write_size_file(item_paths, clips_per_item, outfile, mode, max_num_labels, settings)

    # split up paths/labels list per thread run
    num_images_per_thread_run = settings.num_items_per_thread * settings.num_threads
    paths_per_thread_run = sublist(frame_paths, num_images_per_thread_run)
    labels_per_thread_run = sublist(labels, num_images_per_thread_run)


    count = 0
    writer = tf.python_io.TFRecordWriter(outfile)
    for run_index in range(len(paths_per_thread_run)):

        paths_in_run = paths_per_thread_run[run_index]
        labels_in_run = labels_per_thread_run[run_index]

        tic = time.time()
        debug("Processing %d items for the run." % len(paths_in_run))

        paths_per_thread = sublist(paths_in_run, settings.num_items_per_thread )
        labels_per_thread = sublist(labels_in_run, settings.num_items_per_thread )

        debug("Items scheduled list len : %d." % (len(paths_per_thread)))

        num_threads_in_run = len(paths_per_thread)
        for t in range(num_threads_in_run):
            debug("Frames scheduled for thread #%d : %d." % (t, len(paths_per_thread[t])))
        # start threads
        threads = [[] for _ in range(num_threads_in_run)]
        frames =  [[] for _ in range(num_threads_in_run)]
        for t in range(num_threads_in_run):
            threads[t] = threading.Thread(target=read_item_list_threaded,args=(paths_per_thread[t],frames,t))
            threads[t].start()


        # wait for threads to read
        for t in range(num_threads_in_run):
            threads[t].join()

        for t in range(num_threads_in_run):
            debug("Frames produced  for thread #%d : %d." % (t, len(frames[t])))


        # write the read images to the tfrecord



        for t in range(num_threads_in_run):
            if not frames[t]:
                error("Thread # %d encountered an error." % t)
                exit(1)
            serialize_to_tfrecord(frames[t], labels_per_thread[t], outfile, writer)
            count += len(frames[t])


        info("Processed %d frames, latest %d-sized batch took %s." %
                    (count, sum(list(map(len,paths_per_thread))), elapsed_str(time.time()-tic)))

    writer.close()


def get_item_paths(paths_list, mode):
    info("Generating paths...")
    tic = time.time()
    paths_per_video = []
    if mode == defs.input_mode.image:
        return paths_list
    else:
        for vid_idx in range(len(paths_list)):
            #info("Processing path %d / %d" % (vid_idx+1, len(paths_list)))
            video_path = paths_list[vid_idx]
            video_frame_paths = get_video_frame_paths(video_path)
            paths_per_video.append(video_frame_paths)
    info("Total path generation time: %s " % elapsed_str(time.time() - tic))
    return paths_per_video



def read_item_list_threaded(paths, storage, id):
    for framepath in paths:
        image = read_image(framepath)
        if image is None:
            return
        storage[id].append(image)


def serialize_to_tfrecord( frames, labels, outfil, writer):
    for idx in range(len(frames)):
        frame = frames[idx]
        label = labels[idx]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(settings.raw_image_shape[0]),
            'width': _int64_feature(settings.raw_image_shape[1]),
            'depth': _int64_feature(settings.raw_image_shape[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(frame.tostring())}))
        writer.write(example.SerializeToString())

# read all frames for a video
def get_video_frame_paths(path):

    files = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    files = sorted(files)
    num_files = len(files)

    clips = []
    # generate a number of frame paths from the video path
    if settings.clipframe_mode == defs.clipframe_mode.rand_frames:

        # select frames randomly from the video
        avail_frames = list(range(num_files))
        shuffle(avail_frames)
        avail_frames = avail_frames[:settings.num_frames_per_clip]
        clips.append(avail_frames)

    elif settings.clipframe_mode == defs.clipframe_mode.rand_clips:

        # get <num_clips> random chunks of a consequtive <num_frames> frames
        possible_chunk_start = list(range(num_files - settings.num_frames_per_clip + 1))
        if len(possible_chunk_start) < settings.clip_offset_or_num:
            error("Video %s cannot sustain a number of %d unique %d-frame clips" %
                  (path, settings.clip_offset_or_num, settings.num_frames_per_clip))
        shuffle(possible_chunk_start)
        for _ in range(settings.clip_offset_or_num):
            start_index = possible_chunk_start[-1]
            possible_chunk_start = possible_chunk_start[:-1]
            clip_frames = list(range(start_index, start_index + settings.num_frames_per_clip))
            clips.append(clip_frames)


    elif  settings.clipframe_mode == defs.clipframe_mode.iterative:
        # get all possible video clips.
        start_indexes = list(range(0 , num_files - settings.num_frames_per_clip + 1,
                                   settings.num_frames_per_clip + settings.clip_offset_or_num))
        for s in start_indexes :
            clip_frames = list(range(s,s+settings.num_frames_per_clip ))
            clips.append(clip_frames)

    clip_frame_paths = []
    for clip in clips:
        frame_paths=[]
        for fridx in clip:
            frame_path = os.path.join(path, files[fridx])
            frame_paths.append(frame_path)
        clip_frame_paths.append(frame_paths)
    return clip_frame_paths

 # read image from disk

def read_image(imagepath):
    try:
        image = imread(imagepath)
        debug("Reading image %s" % imagepath)
        # for grayscale images, duplicate
        # intensity to color channels
        if len(image.shape) <= 2:
            image = np.repeat(image[:, :, np.newaxis], 3, 2)
        # drop channels other than RGB
        image = image[:,:,:3]
        #  convert to BGR
        image = image[:, :, ::-1]
        # resize
        image = imresize(image, settings.raw_image_shape)

        # there is a problem if we want to store mean-subtracted images, as we'll have to store a float per pixel
        # => 4 x the space of a uint8 image
        # image = image - mean_image
    except Exception as ex:
        error("Error :" + str(ex))
        error("Error reading image.")
        return None
    return image

# read from tfrecord
def deserialize_from_tfrecord( iterator, images_per_iteration):
    # images_per_iteration :
    images = []
    labels  = []
    for _ in range(images_per_iteration):
        try:
            string_record = next(iterator)
            example = tf.train.Example()
            example.ParseFromString(string_record)
            img_string = (example.features.feature['image_raw']
                          .bytes_list
                          .value[0])
            # height = int(example.features.feature['height']
            #              .int64_list
            #              .value[0])
            # width = int(example.features.feature['width']
            #             .int64_list
            #             .value[0])
            #
            # depth = (example.features.feature['depth']
            #          .int64_list
            #          .value[0])
            label = (example.features.feature['label']
                     .int64_list
                     .value)
            label = list(label)
            label = label[0] if len(label) == 0 else label
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            # watch it : hardcoding preferd dimensions according to the dataset object.
            # it should be the shape of the stored image instead, for generic use
            image = img_1d.reshape((settings.raw_image_shape[0], settings.raw_image_shape[1], settings.raw_image_shape[2]))

            images.append(image)
            labels.append(label)

        except StopIteration:
            break
        except Exception as ex:
            error('Exception at reading image, loading from scratch')
            error(ex)
            error("Error reading tfrecord image.")

    return images, labels



def read_file(inp):
    mode = None
    info("Reading input file [%s] " % (inp))
    max_num_labels = -1
    paths = []
    labels = []
    with open(inp, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            path, label = line.strip().split(" ",1)
            label = [ int(l) for l in label.split() ]
            # document the maximum number of labels
            if len(label) > max_num_labels:
                max_num_labels = len(label)

            if mode is None:
                if path.lower().endswith("." + settings.frame_format.lower()):
                    mode = defs.input_mode.image
                    info("Set input mode to frames from paths-file items suffixes.")
                else:

                    mode = defs.input_mode.video
                    strlen = min(len(path), len(settings.frame_format) + 1)
                    suffix = path[-strlen:]
                    info(
                        "Set input mode to videos since paths-file item suffix [%s] differs from image format [%s]." % (
                        suffix, settings.frame_format))

            if settings.path_prepend_folder is not None:
                path = os.path.join(settings.path_prepend_folder, path)
            paths.append(path)
            labels.append(label)
    return paths, labels, mode, max_num_labels


def shuffle_pair(*args):
    z = list(zip(*args))
    shuffle(z)
    args= zip(*z)
    return args

def shuffle_paths(item_paths, paths, labels, mode):
    info("Shuffling data...")

    if mode == defs.input_mode.image:
        item_paths, labels = shuffle_pair(item_paths, labels)
        return item_paths, labels

    # outer shuffle, of video order
    item_paths, paths, labels = shuffle_pair(item_paths, paths,labels)


    # inner shuffle of frames, if clip frame mode is random frames
    if  settings.clipframe_mode == defs.clipframe_mode.rand_frames:

        for vid_idx in range(len(paths)):
            for clip_idx in range(len(paths[vid_idx])):
                shuffle(paths[vid_idx][clip_idx])
        return paths, labels
    else:
        # here we can only shuffle the clips themselves, not the frames within
        for vid_idx in range(len(paths)):
            shuffle(paths[vid_idx])
        return item_paths, paths, labels

def write(settings):
    # store written data per input file, to print shuffled & validate, later
    framepaths_per_input = []
    for idx in range(len(settings.input_files)):
        inp = settings.input_files[idx]
        item_paths, item_labels, mode, max_num_labels = read_file(inp)

        if mode == defs.input_mode.image:
            if settings.do_shuffle:
                item_paths, item_labels = shuffle_paths(item_paths, None, item_labels, mode)
            paths_to_serialize, labels_to_serialize = item_paths, item_labels
            clips_per_item = None
            framepaths_per_input.append([item_paths, item_labels, None, None, mode])

        elif mode == defs.input_mode.video:
            # generate paths per video
            paths = get_item_paths(item_paths, mode)

            if settings.do_shuffle:
                item_paths, paths, item_labels = shuffle_paths(item_paths, paths, item_labels, mode)
            clips_per_item = [ len(vid) for vid in paths ]

            # flatten: frame, for video in videos for frame in video
            labels_to_serialize = []
            for idx in range(len(item_labels)):
                ll = [item_labels[idx] for clip in paths[idx] for _ in clip]
                labels_to_serialize.extend(ll)
            paths_to_serialize = [ p for video in paths for clip in video for p in clip]
            framepaths_per_input.append([item_paths, item_labels, paths_to_serialize, labels_to_serialize, mode])
        else:
            error("Unknown data type: ",mode)

        if settings.do_serialize:
            tic = time.time()
            output_file = inp + ".tfrecord"
            info("Serializing %s " % (output_file))
            serialize_multithread(item_paths, clips_per_item, paths_to_serialize, labels_to_serialize,
                                  output_file , mode, max_num_labels, settings)
            info("Done serializing %s " % inp)
            info("Total serialization time: %s " % elapsed_str(time.time() - tic))
        info("Done processing input file %s" % inp)

    return framepaths_per_input

# verify the serialization validity
def validate(written_data, settings):


    for index in range(len(settings.input_files)):

        inp = settings.input_files[index]
        info('Validating %s' % inp)

        item_paths, item_labels, paths, labels, mode,  = written_data[index]
        if mode == defs.input_mode.video and not settings.do_serialize:
            error("Cannot validate-only in video mode, as frame selection is not known.")
        if settings.do_shuffle and not settings.do_serialize:
            error("Cannot validate-only with shuffle enabled, as serialization shuffling is not known.")
        if mode == defs.input_mode.image:
            paths = item_paths
            labels = item_labels

        # validate

        num_validate = round(len(paths) * settings.validate_pcnt / 100) if len(paths) >= 10000 else len(paths)
        info("Will validate %d%% of a total of %d items (but at least 10K), i.e. %d items." % (settings.validate_pcnt, len(paths), num_validate))
        sys.stdout.flush()
        progress = ProgressBar(num_validate, fmt=ProgressBar.FULL)
        error_free = True
        idx_list = [ i for i in range(len(paths))]
        shuffle(idx_list)
        idx_list = idx_list[:num_validate]
        idx_list.sort()
        lidx = 0
        testidx = idx_list[lidx]
        iter = tf.python_io.tf_record_iterator(inp + ".tfrecord")
        if not os.path.isfile(inp + ".tfrecord"):
            error("TFRecord file %s does not exist." % (inp + ".tfrecord"))
        for i in range(len(paths)):
            if not i == testidx:
                next(iter)
                continue
            progress()
            frame = read_image(paths[i])
            label = labels[i]

            fframetf, llabeltf = deserialize_from_tfrecord(iter,1)
            frametf = fframetf[0]
            labeltf = llabeltf[0]

            if not np.array_equal(frame , frametf):
                error("Unequal image @ %s" % paths[i])
                error_free = False
            if not label == labeltf:
                error("Unequal label @ %s. Found %d, expected %d" % ( paths[i], label, labeltf))
                error_free = False

            lidx = lidx + 1
            if lidx >= len(idx_list):
                break

            testidx = idx_list[lidx]
        progress.done()
        if not error_free:
            error("errors exist.")
        else:
            info("Validation for %s ok" % (inp + ".tfrecord"))



def write_paths_file(data, settings):
    # write the selected clips / frames
    for i in range(len(data)):
        inp = settings.input_files[i]

        item_paths, item_labels, paths, labels, mode = data[i]

        if settings.do_shuffle:
            # write paths, if they got shuffled
            item_outfile = inp + ".shuffled"
            info("Documenting shuffled video order to %s" % (item_outfile))
            with open(item_outfile,'w') as f:
                for v in range(len(item_paths)):
                    item = item_paths[v]
                    f.write("%s " % item)
                    if type(item_labels[v]) == list:
                        for l in item_labels[v]:
                            f.write("%d " % l)
                    else:
                        f.write("%d" % item_labels[v])
                    f.write("\n")



        clip_info = "" if settings.clipframe_mode == defs.clipframe_mode.rand_frames or \
                          mode == defs.input_mode.image else ".%d.cpv" % settings.clip_offset_or_num
        outfile = "%s%s.%dfpc.%s" % (inp, clip_info, settings.num_frames_per_clip, settings.clipframe_mode)

        if not mode == defs.input_mode.video:
            continue
        info("Documenting selected paths from file %s \n\tto %s" % (inp, os.path.basename(outfile)))
        with open(outfile, "w") as f:
            for path, label in zip(paths, labels):
                f.write("%s %s\n" % (path, " ".join(list(map(str,label)))))


if __name__ == '__main__':
    settings = serialization_settings()
    settings.initialize_from_file(sys.argv)
    # outpaths is either the input frame paths in image mode, or the expanded frame paths in video mode
    written_data = write(settings)
    write_paths_file(written_data, settings)
    if settings.do_validate:
        validate(written_data, settings)



