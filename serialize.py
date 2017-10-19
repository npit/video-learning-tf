import tensorflow as tf
import numpy as np
from random import shuffle, choice
from scipy.misc import imread, imresize, imsave

import logging, time, threading, os, configparser, sys
from os.path import basename
from shutil import copyfile
from utils_ import *
import tqdm
from defs_ import *
import pickle
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

        if self.run_id is None:
            self.run_id = "serialize_%s" % (get_datetime_str())
        else:
            print("Using explicit run id of [%s]" % self.run_id)
        # configure the logs
        logfile = "log_" + self.run_id  + ".log"
        self.logger = CustomLogger()
        CustomLogger.instance = self.logger
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
    for ridx, rpaths in enumerate(paths_per_thread_per_run):
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
                pbar.set_description("Run %d/%d, processed %7d/%7d frames" %
                                     (run_index + 1, len(paths_per_run), count,
                                     len(frame_paths)))
                pbar.update()

    info("Time elapsed for file serialization: %s" % elapsed_str(time.time()-tic))

    writer.close()

def generate_frames_per_video(paths_list, mode):
    tic = time.time()
    paths_per_video = []
    info("Fetching frame paths for %d videos, using %s with %d cpv and %d fpc." % 
         (len(paths_list), settings.clipframe_mode, settings.clip_offset_or_num, settings.num_frames_per_clip))
    with tqdm.tqdm(range(len(paths_list)), ascii=True, total=len(paths_list)) as pbar:
        for vid_idx in range(len(paths_list)):
            video_path = paths_list[vid_idx]
            video_frame_paths = generate_frames_for_video(video_path)
            paths_per_video.append(video_frame_paths)
            pbar.set_description("Processing %-30s" % basename(video_path))
            pbar.update()
    info("Total generation time for %f total paths: %s " % (sum([len(p) for p in paths_per_video]), elapsed_str(time.time()-tic)))
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
            'height': _int64_feature(frame.shape[0]),
            'width': _int64_feature(frame.shape[1]),
            'depth': _int64_feature(frame.shape[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(frame.tostring())}))
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
                error("Undefined generation error stragegy: %s" % settings.generation_error)

        avail_frame_idxs = avail_frame_idxs[:settings.num_frames_per_clip]
        return avail_frame_idxs

def get_random_clips(avail_frame_idxs, settings, path):
        # get <num_clips> random chunks of a consequtive <num_frames> frames
        # the clips may overlap
        # handle videos with too few frames
        num_frames = len(avail_frame_idxs)
        num_frames_missing = settings.num_frames_per_clip - num_frames
        if num_frames_missing > 0:
            message = "Video %s cannot sustain a number of %d fpc, as it has %d frames" % (basename(path), settings.num_frames_per_clip, num_frames)
            if settings.generation_error == defs.generation_error.abort:
                error(message)
            # log the error for later
            settings.logger.add_to_log_storage("generation", (message, path))
            if settings.generation_error == defs.generation_error.compromise:
                # duplicate start frame to match the fpc,
                # duplicate the clip to match the cpv
                avail_frame_idxs = [0 for _ in range(num_frames_missing)] + avail_frame_idxs
                return [avail_frame_idxs for _ in range(settings.clip_offset_or_num)]
            elif settings.generation_error == defs.generation_error.report:
                return []
            else:
                error("Undefined generation error stragegy: %s" % settings.generation_error)


        possible_clip_start = list(range(num_frames - settings.num_frames_per_clip + 1))
        # handle videos that cannot support that many clips
        num_clips_missing = settings.clip_offset_or_num - len(possible_clip_start)
        if num_clips_missing > 0:
            message = "Video %s cannot sustain a number of %d,%d cpv, fpc in, as it has %d frames" % (basename(path), settings.num_frames_per_clip, settings.clip_offset_or_num, num_frames)
            if settings.generation_error == defs.generation_error.abort:
                error(message)
            # log the error for later
            settings.logger.add_to_log_storage("generation", (message, path))
            if settings.generation_error == defs.generation_error.compromise:
                # duplicate clip starts to match the required
                possible_clip_start.extend([choice(possible_clip_start) for _ in range(num_clips_missing)])
                settings.logger.add_to_log_storage("generation",(message, path))
            elif settings.generation_error == defs.generation_error.report:
                return []
            else:
                error("Undefined generation error stragegy: %s" % settings.generation_error)

        shuffle(possible_clip_start)
        possible_clip_start = possible_clip_start[:settings.clip_offset_or_num]
        ret = [list(range(st,st+settings.num_frames_per_clip)) for st in possible_clip_start]
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
                                         in num_frames_missing])
            elif settings.generation_error == defs.generation_error.report:
                return []
            else:
                error("Undefined generation error stragegy: %s" % settings.generation_error)

        clip_start_distance = settings.num_frames_per_clip + settings.clip_offset_or_num
        start_indexes = list(range(0 , num_frames - settings.num_frames_per_clip + 1, clip_start_distance))
        return [list(range(s,s+settings.num_frames_per_clip )) for s in start_indexes]

# generate frames per video, according to the input settings 
def generate_frames_for_video(path):

    files = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
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

def write_serialization(settings):
    # store written data per input file, to print shuffled & validate, later
    framepaths_per_input = []
    errors_per_input = [False for _ in settings.input_files]
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
            paths = generate_frames_per_video(item_paths, mode)
            # check generation status
            stored_log = settings.logger.get_log_storage("generation")
            if stored_log:
                # errors exist, print them
                errors_per_input[idx] = True
                warning("%d generation errors occured, that were resolved with the [%s] strategy:" %
                        (len(stored_log), settings.generation_error))
                for i,logline, _ in enumerate(stored_log):
                    warning("%d/%d: %s" % (i+1, len(logline), logline))
                # handle the errors according to the generation error strategy
                if settings.generation_error == defs.generation_error.report:
                    probl_savefile = "generation_errors_files_%s_%s"
                    with open(probl_savefile % (settings.run_id, get_datetime_str()),"w") as f:
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
                else:
                    error("Generated paths with errors, but error strategy is [%s]" % settings.generation_error)

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
            output_file = inp + ".tfrecord"
            if settings.output_folder is not None:
                output_file = os.path.join(settings.output_folder, basename(output_file))
                if not os.path.exists(settings.output_folder):
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

        inp = settings.input_files[index]

        if errors[index]:
            info("Skipping file %s due to generation errors and stragegy [%s]" % (basename(inp), settings.generation_error))
            continue

        output_file = inp + ".tfrecord"
        if settings.output_folder is not None:
            output_file = os.path.join(settings.output_folder, basename(output_file))
        if not os.path.isfile(output_file):
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

        # validate
        num_validate = round(len(paths) * settings.validate_pcnt / 100) if len(paths) >= 10000 else len(paths)
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
        with tqdm.tqdm(total=num_validate, desc="Validating [%s]" %
                  basename(output_file), ascii=True) as pbar:
            for i in range(len(paths)):
                if not i == testidx:
                    next(iter)
                    continue
                frame, label = read_image(paths[i]), labels[i]

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
            info("Validation for %s completed successfully." % (inp + ".tfrecord"))
    info("Validation completed error-free for all files.")

def write_paths_file(data, errors, settings):
    info("Writing serialization metadata")
    # write the selected clips / frames
    for i in range(len(data)):
        inp = settings.input_files[i]
        if errors[i]:
            info("Skipping file %s due to generation errors and stragegy [%s]" % (basename(inp), settings.generation_error))
            continue

        item_paths, item_labels, paths, labels, mode = data[i]

        if settings.do_shuffle:
            # re-write paths, if they got shuffled, renaming the original
            if settings.output_folder is not None:
                output_file = os.path.join(settings.output_folder,
                                           basename(inp))
            else:
                output_file = inp

            copyfile(inp, output_file + ".unshuffled")
            shuffled_paths_file= output_file + ".shuffled"
            info("Documenting shuffled video order to %s" % (shuffled_paths_file))
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

if __name__ == '__main__':
    settings = serialization_settings()
    settings.initialize_from_file(sys.argv)
    # outpaths is either the input frame paths in image mode, or the expanded frame paths in video mode
    written_data, errors_per_file = write_serialization(settings)
    write_paths_file(written_data, errors_per_file, settings)
    if settings.do_validate:
        info("Validating serialization")
        validate(written_data, errors_per_file, settings)



