#!/usr/bin/env python3

import os, sys, random, configparser
sys.path.append('..')
from utils_ import *
"""
Script to produce a frame list from a collection of videos.
"""
################################

init_file = "config.ini"
# a file containing relative video paths and classes
input_files = []
# frames per video clip chunk
num_frames_per_clip = 16
# number of clips for a video
num_clips_per_video = 1
# in image mode, select random <num_frames_per_clip> frames. In video mode, select a contiguous images chunk
input_mode = defs.input_mode.video
paths_prepend_folder = "path/to/frames"

frame_format = "jpg"

# initialize from file
def initialize_from_file(init_file):
    if init_file is None:
        return
    if not os.path.exists(init_file):
        return
    tag_to_read = "framesprod"
    print("Initializing from file %s" % init_file)
    config = configparser.ConfigParser()
    config.read(init_file)
    if not config[tag_to_read ]:
        error('Expected header [%s] in the configuration file!' % tag_to_read)

    config = config[tag_to_read]


    if config['input_files']:
        input_files =config['input_files']
        input_files = input_files.split(",")
        input_files = list(map(lambda x: x.strip(), input_files))

    # keys and default values
    keys = ['num_frames_per_clip', 'num_clips_per_video', 'input_mode', 'paths_prepend_folder', 'frame_format']

    values = [ eval(k) for k in keys ]
    funcs = [eval for _ in keys]

    for i in range(len(keys)):
        try:
            key = keys[i]
            value = config[key]
            if funcs[i] is not None:
                value = list(map(funcs[i], [value]))
                if len(value) == 1:
                    value = value[0]
            values[i] = value
        except KeyError as k:
            print("Warning: Option %s undefined" % str(k))
            pass

    print("Successfully initialized from file %s" % init_file)
    values .append(input_files)
    return tuple(values)

# read configuration
num_frames_per_clip, num_clips_per_video, input_mode, paths_prepend_folder, frame_format, input_files = initialize_from_file(init_file)

# loop over input files
for inp in input_files:

    video_paths = []
    labels = []
    frame_paths = []
    print("Processing %s" % inp)

    # read paths
    with open(inp, "r") as f:
        for line in f:
            path, label = line.split()
            path = path.strip()
            label = int(label.strip())

            video_paths.append(path)
            labels.append(label)
    print("Read %d , %d paths" % (len(video_paths), len(labels)))
    # make the list for the video frames, for each clip per video
    fr_list = range(1, num_frames_per_clip + 1)


    # for each path count the number of files
    for p in range(len(video_paths)):

        path = video_paths[p]
        print("%d/%d : %s" % (1+p, len(video_paths), path))

        truepath = path
        if paths_prepend_folder is not None:
            truepath = os.path.join(paths_prepend_folder, path)
        if not os.path.exists(truepath):
            error("Path [%s] does not exist!" % path)

        num_files = len(os.listdir(truepath))

<<<<<<< HEAD
        # generate a number of frame paths from the video path
        if input_mode == defs.input_mode.image:
            # select frames randomly from the video
            avail_frames = list(range(num_files))
            random.shuffle(avail_frames)
            avail_frames = avail_frames[:num_frames_per_clip]
            frame_paths.append([avail_frames])
        elif input_mode == defs.input_mode.video:
            # get <num_clips> random chunks of a consequtive <num_frames> frames
            frame_paths.append([])
            possible_chunk_start = list(range(num_files - num_frames_per_clip + 1))
            if len(possible_chunk_start) < num_clips_per_video:
                error("Video %s cannot sustain a number of %d unique %d-frame clips" % (truepath, num_clips_per_video, num_frames_per_clip))
            random.shuffle(possible_chunk_start)
            for _ in range(num_clips_per_video):
                start_index = possible_chunk_start[-1]
                possible_chunk_start=possible_chunk_start[:-1]
                clip_frames = list(range(start_index, start_index  + num_frames_per_clip))
                frame_paths[-1].append(clip_frames)

    # write
    outfile = "%s.%s.%d.cpv.%d.fpv" % (inp, defs.input_mode.str(input_mode),num_clips_per_video, num_frames_per_clip)
    with open(outfile,"w") as f:
        # for each video
        for idx in range(len(frame_paths)):
            frames = frame_paths[idx]
            video_label = labels[idx]
            videopath = video_paths[idx]
            for clip_idx in range(len(frames)):
                for fidx in range(len(frames[clip_idx])):
                    frame = frames[clip_idx][fidx]
                    frame = "%s/%s.%04d.%s" % (videopath ,videopath , 1+frame, frame_format)
                    frame = os.path.join(paths_prepend_folder , frame)
                    f.write("%s %d\n" % (frame, video_label))



