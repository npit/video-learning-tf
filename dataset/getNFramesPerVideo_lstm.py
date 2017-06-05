#!/usr/bin/env python3

import os, random
from utils_ import *
"""
Script to produce a frame list from a collection of videos
"""
################################

init_file = "config.ini"
# a file containing relative video paths and classes
input_files = [
    "/home/nik/uoa/msc-thesis/implementation/dataset/donahue_splits/ucf101_split1_trainVideos.txt",
    "/home/nik/uoa/msc-thesis/implementation/dataset/donahue_splits/ucf101_split1_testVideos.txt"
]
num_frames_per_video = 16
# in image mode, select random <num_frames_per_video> frames. In video mode, select a contiguous images chunk
input_mode = defs.input_mode.video
paths_prepend_folder = "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames"


frame_format = "jpg"

for inp in input_files:

    video_paths = []
    labels = []
    frame_paths = []
    print("Processing %s" % inp)
    with open(inp, "r") as f:
        for line in f:
            path, label = line.split()
            path = path.strip()
            label = int(label.strip())

            video_paths.append(path)
            labels.append(label)
    print("Read %d , %d paths" % (len(video_paths), len(labels)))
    fr_list = range(1, num_frames_per_video + 1)

    # for each path count the number of files
    for p in range(len(video_paths)):

        path = video_paths[p]
        print("%d/%d : %s" % (p, len(video_paths), path))
        truepath = path
        if paths_prepend_folder is not None:
            truepath = os.path.join(paths_prepend_folder, path)
        if not os.path.exists(truepath):
            error("Path [%s] does not exist!" % path)

        num_files = len(os.listdir(truepath))

        if input_mode == defs.input_mode.image:
            # select frames randomly
            avail_frames = list(range(num_files))
            random.shuffle(avail_frames)
            avail_frames = avail_frames[:num_frames_per_video]
        elif input_mode == defs.input_mode.video:
            # get a random chunk of <num_frames> frames
            possible_chunk_start = list(range(num_files - num_frames_per_video + 1))
            start_index = random.choice(possible_chunk_start)
            avail_frames = list(range(start_index, start_index  + num_frames_per_video))
        frame_paths.append(avail_frames)

    # write
    if paths_prepend_folder is None:
        paths_prepend_folder = ""
    else:
        paths_prepend_folder = paths_prepend_folder + "/"

    outfile = "%s.%s.%d.fpv" % (inp, defs.input_mode.str(input_mode),num_frames_per_video)
    with open(outfile,"w") as f:
        for idx in range(len(frame_paths)):
            frames = frame_paths[idx]
            label = labels[idx]
            videopath = video_paths[idx]
            for fidx in range(len(frames)):
                frame = frames[fidx]
                frame = "%s/%s.%04d.%s" % (videopath ,videopath , frame, frame_format)
                frame = paths_prepend_folder + frame
                f.write("%s %d\n" % (frame, label))


