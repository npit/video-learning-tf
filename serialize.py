import tensorflow as tf
import itertools
import numpy as np
import pandas as pd
from random import shuffle, choice, random, seed
from scipy.misc import imread, imresize, imsave

import logging, time, threading, os, configparser, sys
from os.path import basename, exists, join, isfile
from shutil import copyfile, move
from utils_ import *
import tqdm
from defs_ import *
import pickle
import yaml
from parse_opts import *
import string
'''
Script for production of training / testing data collections and serialization to tf.record files.
'''
class serialization_settings:
    init_file = "config.ini"
    run_id = None

    # necessary config. variables
    input_files = []

    # defaults
    path_prepend_folder = None
    output_folder = None
    num_threads = 4
    num_items_per_thread = 500
    num_frames_per_clip = 16
    raw_image_shape = (240, 320, 3)
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
        if not exists(self.init_file):
            error("Initialization file [%s] does not exist" % self.init_file)
            return
        tag_to_read = "serialize"
        print("Initializing from file %s" % self.init_file)
        if self.init_file.endswith(".ini"):
            error("Ini files deprecated")
        
        if ".yml" in self.init_file:
            with open(self.init_file,"r") as f:
                config = yaml.load(f)[tag_to_read]
            self.output_folder = config['output_folder']
            self.path_prepend_folder = config['path_prepend_folder']
            self.input_files = [ x.strip() for x in parse_seq(config['input_files'])]
            self.run_id = config['run_id'].strip()
            self.num_threads = int(config['num_threads'])
            self.num_items_per_thread = int(config['num_items_per_thread'])
            self.raw_image_shape = parse_seq(config['raw_image_shape'])
            self.clip_offset_or_num = int(config['clip_offset_or_num'])
            self.num_frames_per_clip = int(config['num_frames_per_clip'])
            self.clipframe_mode = defs.check(config['clipframe_mode'], defs.clipframe_mode)
            self.generation_error = defs.check(config['generation_error'], defs.generation_error)
            self.do_shuffle = config['do_shuffle']
            self.do_serialize = config['do_serialize']
            self.do_validate = config['do_validate']
            self.frame_format = config['frame_format'].strip()
            self.logging_level = config['logging_level'].strip()
            loglevels = ['logging.' + x for x in ['INFO','DEBUG','WARN']]
            if not self.logging_level in loglevels:
                error("Invalid logging level: [%s]" % (self.logging_level))
        else:
            error("Need a yml initialization file")



        if self.run_id is None or not self.run_id:
            self.run_id = "serialize_%s" % (get_datetime_str())
        else:
            print("Using explicit run id of [%s]" % self.run_id)
        # configure the logs
        self.email_notify = config['email_notify']
        if self.email_notify:
            self.email_notify = prep_email(self.email_notify)
        self.logfile = "log_" + self.run_id + ".log"
        self.logger = CustomLogger()
        CustomLogger.instance = self.logger
        self.logger.configure_logging(self.logfile, self.logging_level, self.email_notify)


        if 'seed' in config:
            try:
                self.seed = float(config['seed'])
            except Exception as ex:
                print(ex)
                error("Invalid seed value: %s - numeric expected" % str(self.seed))
                exit(1)
            info("Using supplied seed: %f" % self.seed)
        else:
            self.seed = random()
            info("Using randomized seed: %f" % self.seed)
        seed(self.seed)
        info("Starting serialization run: [%s]" % self.run_id)
        info("Successfully initialized from file %s" % self.init_file)



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
        if clips_per_item is not None:
            cpv_str = [(len(list(g)), k) for k,g in itertools.groupby(clips_per_item)]
        else:
            cpv_str = str(clips_per_item)
        f.write("cpi\t%s\n" % cpv_str)
        f.write("fpc\t%s\n" % str(settings.num_frames_per_clip))
        f.write("labelcount\t%s\n" % str(max_num_labels))

def serialize_multithread(item_paths, clips_per_item, frame_paths, labels, outfile, mode, max_num_labels, settings):

    write_size_file(item_paths, clips_per_item, outfile, mode, max_num_labels, settings)

    # precompute 
    num_images_per_run = settings.num_items_per_thread * settings.num_threads
    paths_per_run = sublist(frame_paths, num_images_per_run )
    labels_per_run = sublist(labels, num_images_per_run)
    paths_per_thread_per_run = [sublist(paths, settings.num_items_per_thread) for paths in paths_per_run]
    labels_per_thread_per_run = [sublist(lbls, settings.num_items_per_thread)
                                 for lbls in labels_per_run]
    total_thread_runs = sum([len(pt) for pt in paths_per_thread_per_run])

    # print schedule
    info("Serialization schedule:")
    max_print = 8
    run_info = list(enumerate(paths_per_thread_per_run))
    if len(run_info) > max_print:
        run_info = run_info[:int(max_print/2)] + [(None,None)] + run_info[-int(max_print/2):]
    for ridx, rpaths in run_info:
        if ridx is None:
            info(".......")
            continue
        run_msg = "Run %d/%d {" % (1+ridx, len(paths_per_thread_per_run))
        for tidx, tpaths in enumerate(rpaths):
            run_msg += " t%d:%d" % (tidx, len(tpaths))
        run_msg +=" }"
        info(run_msg)

    tic = time.time()
    count = 0
    writer = tf.python_io.TFRecordWriter(outfile)
    with tqdm.tqdm(total=total_thread_runs, ascii=True) as pbar:
        for run_index in range(len(paths_per_run)):
            paths_per_thread = paths_per_thread_per_run[run_index]
            labels_per_thread = labels_per_thread_per_run[run_index]

            num_threads_in_run = len(paths_per_thread)
            # make thread result containers
            threads = [[] for _ in range(num_threads_in_run)]
            frames =  [[] for _ in range(num_threads_in_run)]
            # start threads
            for t in range(num_threads_in_run):
                threads[t] = threading.Thread(target=read_item_list_threaded, args=(paths_per_thread[t], frames, t, settings))
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
                serialize_frames_to_tfrecord(frames[t], labels_per_thread[t], writer)
                count += len(frames[t])
                pbar.set_description("Run %d/%d, processed %7d/%7d frames" %
                                     (run_index + 1, len(paths_per_run), count,
                                     len(frame_paths)))
                pbar.update()

    info("Time elapsed for file serialization: %s" % elapsed_str(tic))

    writer.close()

def generate_frames_per_video(paths_list, mode, settings):
    tic = time.time()
    paths_per_video = []
    info("Reading raw frames from folder:[%s]" % settings.path_prepend_folder)
    info("Fetching frame paths for %d videos, using %s with %d cpv and %d fpc." %
         (len(paths_list), settings.clipframe_mode, settings.clip_offset_or_num, settings.num_frames_per_clip))
    with tqdm.tqdm(range(len(paths_list)), ascii=True, total=len(paths_list)) as pbar:
        for vid_idx in range(len(paths_list)):
            video_path = paths_list[vid_idx]
            video_frame_paths = generate_frames_for_video(video_path, settings)
            paths_per_video.append(video_frame_paths)
            pbar.set_description("Processing %-30s" % basename(video_path))
            pbar.update()
    total_num_paths = int(sum([len(p) for p in paths_per_video]))
    info("Total generation time for a total of %d clips: %s " % (total_num_paths, elapsed_str(tic)))
    return paths_per_video

def read_item_list_threaded(paths, storage, id, settings):
    for framepath in paths:
        image = read_image(framepath, settings)
        if image is None:
            return
        storage[id].append(image)

def serialize_frames_to_tfrecord( frames, labels, writer):
    for idx in range(len(frames)):
        frame = frames[idx]
        label = labels[idx]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(frame.shape[0]),
            'width': _int64_feature(frame.shape[1]),
            'depth': _int64_feature(frame.shape[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(frame.tostring())}))
        writer.write(example.SerializeToString())

def serialize_vectors_to_tfrecord( vectors, labels, writer):
    dim = vectors.shape[-1]
    for idx in range(len(vectors)):
        vector = vectors[idx]
        label = labels[idx]
        example = tf.train.Example(features=tf.train.Features(feature={
            'dimension': _int64_feature(dim),
            'label': _int64_feature(label),
            'vector_raw': _bytes_feature(vector.tostring())}))
        writer.write(example.SerializeToString())

def get_random_frames(avail_frame_idxs, settings, path):
        # select frames randomly from the video
        avail_frame_idxs = shuffle(avail_frame_idxs)
        num_frames = len(avail_frame_idxs)
        # handle videos with too few frames
        num_frames_missing = settings.num_frames_per_clip - num_frames
        if num_frames_missing > 0:
            message = "Attempted to get a %d-framed clip from video %s which has %d frames." % (settings.num_frames_per_clip, basename(path), num_frames)
            if settings.generation_error == defs.generation_error.abort:
                error(message)
            # log the error for later
            settings.logger.add_to_log_storage("generation", (message, path))
            if settings.generation_error == defs.generation_error.compromise:
                # to compromise, just duplicate random frames until we're good
                avail_frame_idxs.extend([choice(avail_frame_idxs) for _
                                         in range(num_frames_missing)])
            elif settings.generation_error == defs.generation_error.report:
                return []
            else:
                error("Undefined generation error strategy: %s" % settings.generation_error)

        avail_frame_idxs = avail_frame_idxs[:settings.num_frames_per_clip]
        return avail_frame_idxs

def get_random_clips(avail_frame_idxs, settings, path):
        # get <num_clips> random chunks of a consequtive <num_frames> frames
        # the clips may overlap
        # handle videos with too few frames
        num_frames = len(avail_frame_idxs)
        if num_frames == 0:
            error("No frames for path [%s]" % path)
        num_frames_missing = settings.num_frames_per_clip - num_frames
        if num_frames_missing > 0:
            message = "Video %s cannot sustain a number of %d fpc, as it has %d frames" % (basename(path), settings.num_frames_per_clip, num_frames)
            debug(message)
            if settings.generation_error == defs.generation_error.abort:
                error(message)
            # log the error for later
            settings.logger.add_to_log_storage("generation", (message, path))
            if settings.generation_error == defs.generation_error.compromise:
                # duplicate start frame to match the fpc,
                avail_frame_idxs = [0 for _ in range(num_frames_missing)] + avail_frame_idxs
                # cannot match the cpv either way - have to construct clip here, since.
                ret = [avail_frame_idxs for _ in range(settings.clip_offset_or_num)]
                return ret
            elif settings.generation_error == defs.generation_error.report:
                # pass for now, to report missing cpv on the following check, too.
                pass
            else:
                error("Undefined generation error strategy: %s" % settings.generation_error)


        possible_clip_start = list(range(num_frames - settings.num_frames_per_clip + 1))
        # handle videos that cannot support that many clips
        num_clips_missing = settings.clip_offset_or_num - len(possible_clip_start)
        if num_clips_missing > 0:
            message = "Video %s cannot sustain a number of %d cpv as it has %d frames" % (basename(path), settings.clip_offset_or_num, num_frames)
            debug(message)
            if settings.generation_error == defs.generation_error.abort:
                error(message)
            # log the error for later
            settings.logger.add_to_log_storage("generation", (message, path))
            if settings.generation_error == defs.generation_error.compromise:
                # duplicate clip starts to match the required
                possible_clip_start.extend([choice(possible_clip_start) for _ in range(num_clips_missing)])
            elif settings.generation_error == defs.generation_error.report:
                return []
            else:
                error("Undefined generation error strategy: %s" % settings.generation_error)

        # random clip selection ensuring frame coverage
        debug("Random clip selection out of %d possible clip starts" % (len(possible_clip_start)))
        clip_starts = []
        curr_possible_clip_starts = possible_clip_start.copy()
        for _ in range(settings.clip_offset_or_num):
            # select clip start
            start = choice(curr_possible_clip_starts)
            clip_starts.append(start)
            # remove previous clip frames
            for i in range(start - settings.num_frames_per_clip + 1, start + settings.num_frames_per_clip):
                if i in curr_possible_clip_starts: curr_possible_clip_starts.remove(i)
            # if none left, reset
            if not curr_possible_clip_starts:
                curr_possible_clip_starts = possible_clip_start.copy()

        ret = [list(range(st,st+settings.num_frames_per_clip)) for st in clip_starts]
        return ret

def get_sequential_clips(avail_frame_idxs, settings, path):
        # get all possible video clips spaced by <clip_offset_or_num> frames
        num_frames = len(avail_frame_idxs)
        num_frames_missing = settings.num_frames_per_clip - num_frames
        if num_frames_missing > 0:
            message = "Attempted to get %d-framed sequential clips from video %s which has %d frames." % (settings.num_frames_per_clip, basename(path), num_frames)
            if settings.generation_error == defs.generation_error.abort:
                error(message)
            # log the error for later
            settings.logger.add_to_log_storage("generation",(message,path))
            if settings.generation_error == defs.generation_error.compromise:
                # to compromise, just duplicate random frames until we're good
                avail_frame_idxs.extend([choice(avail_frame_idxs) for _
                                         in range(num_frames_missing)])
            elif settings.generation_error == defs.generation_error.report:
                return []
            else:
                error("Undefined generation error strategy: %s" % settings.generation_error)

        clip_start_distance = settings.num_frames_per_clip + settings.clip_offset_or_num
        start_indexes = list(range(0 , num_frames - settings.num_frames_per_clip + 1, clip_start_distance))
        return [list(range(s,s+settings.num_frames_per_clip )) for s in start_indexes]

# generate frames per video, according to the input settings 
def generate_frames_for_video(path, settings):

    debug("Generating frames for video [%s]" % path)
    files = [ f for f in os.listdir(path) if isfile(join(path,f))]
    num_frames = len(files)
    avail_frame_idxs = list(range(num_frames))

    clips = []
    # generate a number of frame paths from the video path
    if settings.clipframe_mode == defs.clipframe_mode.rand_frames:
        clips = get_random_frames(avail_frame_idxs, settings, path)

    elif settings.clipframe_mode == defs.clipframe_mode.rand_clips:
        clips = get_random_clips(avail_frame_idxs, settings, path)

    elif  settings.clipframe_mode == defs.clipframe_mode.iterative:
        clips = get_sequential_clips(avail_frame_idxs, settings, path)

    clip_frame_paths = []
    files = sorted(files)
    debug("Writing a total of %d frames for %d clips" % (sum([len(c) for c in clips]),len(clips)))
    for clip in clips:
        frame_paths=[]
        for fridx in clip:
            frame_path = join(path, files[fridx])
            frame_paths.append(frame_path)
        clip_frame_paths.append(frame_paths)
    return clip_frame_paths

 # read image from disk
def read_image(imagepath, settings):
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
        if settings.raw_image_shape is not None:
            image = imresize(image, settings.raw_image_shape)

        # there is a problem if we want to store mean-subtracted images, as we'll have to store a float per pixel
        # => 4 x the space of a uint8 image
        # image = image - mean_image
    except Exception as ex:
        error("Error :" + str(ex))
        error("Error reading image.")
        return None
    return image

def deserialize_vector(iterator, num_vectors):
    vectors, labels = [], []
    for _ in range(num_vectors):
        try:
            string_record = next(iterator)
            example = tf.train.Example()
            example.ParseFromString(string_record)
            vec_string = (example.features.feature['vector_raw']
                .bytes_list
                .value[0])
            dim = int(example.features.feature['dimension']
                         .int64_list
                         .value[0])
            label = (example.features.feature['label']
                     .int64_list
                     .value)
            label = list(label)
            label = label[0] if len(label) == 0 else label
            vector = np.fromstring(vec_string, dtype=np.float32)
            if dim != len(vector):
                error("Deserialized vector length %d but dimension stored is %d." % (len(vector), dim))

            vectors.append(vector)
            labels.append(label)

        except StopIteration:
            break
        except Exception as ex:
            warning(ex)
            error("Error reading tfrecord vector.")

    return vectors, labels

# read from tfrecord
def deserialize_from_tfrecord(iterator, images_per_iteration):
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
            # reshape according to the stored dimensions
            image = img_1d.reshape((height, width, depth))

            images.append(image)
            labels.append(label)

        except StopIteration:
            break
        except Exception as ex:
            error('Exception at reading image, loading from scratch')
            error(ex)
            error("Error reading tfrecord image.")

    return images, labels

def read_file(inp, settings):
    mode = None
    if settings.path_prepend_folder is not None:
        info("Prepending path:[%s]" % settings.path_prepend_folder)
    max_num_labels = -1
    paths = []
    labels = []
    with open(inp, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = line.strip()
            path, label = line.split(" ",1)
            # if all elements are numeric, it's a features file
            if not any([x in string.ascii_letters for x in path]):
                mode = defs.input_mode.vectors
                info("Set input mode to [%s] due to non-letter path value." % (mode))
                break
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
                        "Set input mode to videos since the first item's suffix: [%s] in the paths-file differs from specified image format [%s]." % (
                        suffix, settings.frame_format))

            if settings.path_prepend_folder is not None:
                path = join(settings.path_prepend_folder, path)
            paths.append(path)
            labels.append(label)
    return paths, labels, mode, max_num_labels

def shuffle_pair(*args):
    z = list(zip(*args))
    shuffle(z)
    args= zip(*z)
    return args

def shuffle_paths(item_paths, paths, labels, mode, settings):
    info("Shuffling data.")

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

def check_cpv_per_item(paths_per_item, items_list, settings):
    # catch any item non-comforming to the cpv
    erratic_items = [i for (i,p) in enumerate(paths_per_item) if len(p) != settings.clip_offset_or_num]
    if any(erratic_items):
        for e in erratic_items:
            item, paths = items_list[e], paths_per_item[e]
            warning("Item %d/%d : %s has cpv of len %d:" % (e+1, len(paths_per_item),item,len(paths)))
            for p in paths:
                warning(p)
        error("Erratic item(s) encountered")

def write_serialization(settings):
    # store written data per input file, to print shuffled & validate, later
    framepaths_per_input = []
    errors_per_input = [False for _ in settings.input_files]
    for idx in range(len(settings.input_files)):
        inp = settings.input_files[idx]
        info("Reading input file %d/%d: [%s] " % (idx+1, len(settings.input_files), inp))
        item_paths, item_labels, mode, max_num_labels = read_file(inp, settings)
        if mode == defs.input_mode.vectors:
            input_file_and_sidx, ids, labels, outfile = serialize_ascii(inp, settings)
            framepaths_per_input.append((input_file_and_sidx, labels, ids, None, mode))
            continue

        if mode == defs.input_mode.image:
            if settings.do_shuffle:
                item_paths, item_labels = shuffle_paths(item_paths, None, item_labels, mode, settings)
            paths_to_serialize, labels_to_serialize = item_paths, item_labels
            clips_per_item = None
            framepaths_per_input.append([item_paths, item_labels, None, None, mode])

        elif mode == defs.input_mode.video:
            # generate paths per video
            paths = generate_frames_per_video(item_paths, mode, settings)
            # check generation status
            stored_log = settings.logger.get_log_storage("generation")
            if stored_log:
                # errors exist, print them
                errors_per_input[idx] = True
                warning("%d generation errors occured, that were resolved with the [%s] strategy:" %
                        (len(stored_log), settings.generation_error))
                for i,(logline, _) in enumerate(stored_log):
                    warning("%d/%d: %s" % (i+1, len(stored_log), logline))
                # handle the errors according to the generation error strategy
                if settings.generation_error == defs.generation_error.report:
                    probl_savefile = "generation_errors_files_%s_%s" % (settings.run_id, get_datetime_str())
                    with open(probl_savefile,"w") as f:
                        for _, problematic_file in stored_log:
                            f.write(problematic_file + "\n")
                    info("Writing problematic files in %s" % probl_savefile)
                    info("Omitting serialization due to generation error setting [%s]." % defs.generation_error.report)

                    # clear generation logs
                    settings.logger.clear_log_storage("generation")
                    continue
                elif settings.generation_error == defs.generation_error.compromise:
                    # errors were fixed on the fly during generation
                    settings.logger.clear_log_storage("generation")
                    errors_per_input[idx] = False
                else:
                    error("Generated paths with errors, but error strategy is [%s]" % settings.generation_error)

            check_cpv_per_item(paths, item_paths, settings)
            if settings.do_shuffle:
                item_paths, paths, item_labels = shuffle_paths(item_paths, paths, item_labels, mode, settings)
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
            output_file = inp + ".tfrecord"
            if settings.output_folder is not None:
                output_file = join(settings.output_folder, basename(output_file))
                if not exists(settings.output_folder):
                    os.makedirs(settings.output_folder)
            info("Serializing to %s " % (output_file))
            serialize_multithread(item_paths, clips_per_item, paths_to_serialize, labels_to_serialize,
                                  output_file, mode, max_num_labels, settings)
            info("Done serializing %s " % inp)
        info("Done processing input file %s" % inp)

    return framepaths_per_input, errors_per_input
# verify the serialization validity
def validate(written_data, errors, settings):

    for index in range(len(settings.input_files)):
        tic = time.time()
        inp = settings.input_files[index]
        if errors[index]:
            info("Skipping file %s due to generation errors and strategy [%s]" % (basename(inp), settings.generation_error))
            continue

        output_file = inp + ".tfrecord"
        if settings.output_folder is not None:
            output_file = join(settings.output_folder, basename(output_file))
        if not isfile(output_file):
            error("TFRecord file %s does not exist." % output_file)

        info('Validating %s' % output_file)

        item_paths, item_labels, paths, labels, mode,  = written_data[index]
        if mode == defs.input_mode.video and not settings.do_serialize:
            error("Cannot validate-only in video mode, as frame selection is not known.")
        if settings.do_shuffle and not settings.do_serialize:
            error("Cannot validate-only with shuffle enabled, as serialization shuffling is not known.")
        if mode == defs.input_mode.image:
            paths = item_paths
            labels = item_labels
        if mode == defs.input_mode.vectors:
            _, shuffle_idx = item_paths

        num_validate = round(len(paths) * settings.validate_pcnt / 100) if len(paths) >= 10000 else len(paths)

        # validate
        info("Will validate %d%% of a total of %d items (but at least 10K), i.e. %d items." % (settings.validate_pcnt, len(paths), num_validate))
        sys.stdout.flush()
        error_free = True
        idx_list = [ i for i in range(len(paths))]
        shuffle(idx_list)
        idx_list = idx_list[:num_validate]
        idx_list.sort()
        lidx = 0
        testidx = idx_list[lidx]
        iter = tf.python_io.tf_record_iterator(output_file)
        vectors = None
        with tqdm.tqdm(total=num_validate, desc="Validating [%s]" % basename(output_file), ascii=True) as pbar:
            for i in range(len(paths)):
                if not i == testidx:
                    next(iter)
                    continue

                if mode == defs.input_mode.vectors:
                    if vectors is None:
                        vectors, labels, maxlabels = read_vectors(inp)
                        if settings.do_shuffle:
                            vectors = vectors[shuffle_idx]
                            labels = [labels[s] for s in shuffle_idx]
                    dvectors, dlabels = deserialize_vector(iter, 1)
                    while type(dlabels) == list and len(dlabels) == 1: dlabels = dlabels[0]
                    if not np.array_equal(dvectors[0] , vectors[i]):
                        error("Unequal image @ idx %s : (%s)(%s)" % (str(i), str(dvectors[0][:10]), str(vectors[i][:10])))
                        error_free = False
                    if not dlabels == labels[i]:
                        error("Unequal label @ %s. Found %s, expected %s" % ( paths[i], str(dlabels), str(labels[i])))
                        error_free = False
                else:
                    frame, label = read_image(paths[i], settings), labels[i]
                    fframetf, llabeltf = deserialize_from_tfrecord(iter,1)
                    frametf, labeltf = fframetf[0], llabeltf[0]

                    if not np.array_equal(frame , frametf):
                        error("Unequal image @ %s" % paths[i])
                        error_free = False
                    if not label == labeltf:
                        error("Unequal label @ %s. Found %d, expected %d" % ( paths[i], label, labeltf))
                        error_free = False

                lidx = lidx + 1
                pbar.update()
                if lidx >= len(idx_list):
                    break

                testidx = idx_list[lidx]

        if not error_free:
            error("errors exist.")
        else:
            info("Validation for %s completed successfully in %s." % (basename(inp) + ".tfrecord", elapsed_str(tic)))
    info("Validation completed error-free for all files.")

def write_paths_file(data, errors, settings):
    info("Writing serialization metadata")
    # write the selected clips / frames
    for i in range(len(data)):
        tic = time.time()
        inp = settings.input_files[i]
        if errors[i]:
            info("Skipping file %s due to generation errors and strategy [%s]" % (basename(inp), settings.generation_error))
            continue

        item_paths, item_labels, paths, labels, mode = data[i]
        if settings.output_folder is not None:
            output_file = join(settings.output_folder, basename(inp))
        else:
            output_file = inp

        if settings.do_shuffle:
            # re-write paths, if they got shuffled, renaming the original
            shuffled_paths_file= output_file + ".shuffled"
            info("Documenting shuffled video order to %s" % (shuffled_paths_file))

            if mode == defs.input_mode.vectors:
                with open(shuffled_paths_file,'w') as f:
                    for item_id, label in zip(item_labels,paths):
                        f.write("%s %s\n" % (item_id, str(label)))
            else:
                copyfile(inp, output_file + ".unshuffled")
                with open(shuffled_paths_file,'w') as f:
                    for v in range(len(item_paths)):
                        item = item_paths[v]
                        f.write("%s " % item)
                        if type(item_labels[v]) == list:
                            for l in item_labels[v]:
                                f.write("%d " % l)
                        else:
                            f.write("%d" % item_labels[v])
                        f.write("\n")
        else:
            # if not shuffle, copy the paths in the same name, if an output folder is defined
            if settings.output_folder is not None:
                copyfile(inp, output_file)

        if mode == defs.input_mode.vectors:
            info("Will not write clip information, as input is vectors")
            info("Possible TODO is for input frame vectors")
            pass
        else:
            clip_info = "" if settings.clipframe_mode == defs.clipframe_mode.rand_frames or \
                              mode == defs.input_mode.image else ".%d.cpv" % settings.clip_offset_or_num
            frame_info = "" if mode == defs.input_mode.image else ".%d.fpc" % settings.num_frames_per_clip
            clipframe_mode_info = "" if mode == defs.input_mode.image else ".%s.cfm" % settings.clipframe_mode
            outfile = "%s%s%s%s" % (output_file, clip_info, frame_info, clipframe_mode_info)

            if not mode == defs.input_mode.video:
                continue
            info("Documenting selected clip/frame/... info to %s" % (basename(outfile)))
            with open(outfile, "w") as f:
                for path, label in zip(paths, labels):
                    f.write("%s %s\n" % (path, " ".join(list(map(str,label)))))

def read_vectors(input_file):

    data = pd.read_csv(input_file, header=None, delimiter=" ").values
    vectors, labels, max_num_labels = None, None, 1
    for i in range(len(data)):
        feature_vector, labels_vector = data[i][0], data[i][-1]
        row = np.asarray(feature_vector.split(","),np.float32)
        if type(labels_vector) is not int:
            print(i,":",labels_vector)
            labels_vector = np.asarray(labels_vector.split(","),np.int32)
        dim = len(row)
        if i == 0:
            vectors = np.ndarray((0, dim), np.float32)
            labels = []
            stored_dim = dim
        if vectors.shape[-1] != stored_dim:
            error("Inconsistent dimension: Encountered dim: %d at line %d, had stored %d." % (dim, i+1, stored_dim))
        vectors = np.vstack((vectors, row))
        labels.append(labels_vector)
        if type(labels_vector) != int:
            max_num_labels = max(max_num_labels, len(labels_vector))
    # return the read data
    return vectors, labels, max_num_labels


def serialize_ascii(input_file, settings):

    info("Reading existing features from file: [%s]" % input_file)
    feature_file, ids_file = input_file, input_file + ".ids"

    # output folder
    if settings.output_folder:
        outfile = join(settings.output_folder, basename(feature_file) + ".tfrecord")
        if not exists(settings.output_folder):
            info("Creating output folder [%s]" % settings.output_folder)
            os.mkdir(settings.output_folder)
    else:
        outfile = feature_file + ".tfrecord"

    # convert string to numpy vector
    vectors, labels, max_num_labels = read_vectors(feature_file)
    ids = [ line.split()[0] for line in  read_file_lines(ids_file)]

    if settings.do_shuffle:
        info("Shuffling vector features, random seed: [%s]" % str(settings.seed))
        shuffle_idx = np.arange(len(vectors))
        np.random.shuffle(shuffle_idx)
        shuffle_idx = np.ndarray.tolist(shuffle_idx)
        vectors = vectors[shuffle_idx]
        labels = [labels[i] for i in shuffle_idx]
        ids = [ids[i] for i in shuffle_idx]
    else:
        shuffle_idx = None
    info("Serializing existing features to file: [%s]" % outfile)
    write_size_file(vectors, [1 for _ in vectors], outfile, defs.input_mode.vectors, max_num_labels, settings)
    writer = tf.python_io.TFRecordWriter(outfile)
    serialize_vectors_to_tfrecord(vectors, labels, writer)
    info("Done serializing to file: [%s]" % outfile)
    return ((input_file, shuffle_idx), ids, labels, outfile)


def main():
    settings = serialization_settings()
    settings.initialize_from_file(sys.argv)
    # outpaths is either the input frame paths in image mode, or the expanded frame paths in video mode
    written_data, errors_per_file = write_serialization(settings)
    write_paths_file(written_data, errors_per_file, settings)
    if settings.do_validate:
        info("Validating serialization")
        validate(written_data, errors_per_file, settings)
    # move log and config files to output directory, if specified and serialization happened
    if settings.output_folder is not None and settings.do_serialize and all([not e for e in errors_per_file]):
        copyfile(settings.logfile, join(settings.output_folder, basename(settings.logfile)))
        copyfile(settings.init_file, join(settings.output_folder, basename(settings.init_file)))
    info("Serialization complete", email = True)


if __name__ == '__main__':
    main()
