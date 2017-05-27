import os, pickle
from random import shuffle

train, val = range(2)
phase = train
videos_in_phase_files = [[], []]
videos_in_phase_files[train] = "/home/nik/uoa/msc-thesis/implementation/caffe-lisa-anne/lisa-caffe-public/examples/LRCN_activity_recognition/ucf101_split1_trainVideos.txt"
videos_in_phase_files[val] = "/home/nik/uoa/msc-thesis/implementation/caffe-lisa-anne/lisa-caffe-public/examples/LRCN_activity_recognition/ucf101_split1_testVideos.txt"
frames_directory = "/home/nik/uoa/msc-thesis/datasets/ready_data_DonahuePaper/frames/"
num_frames_per_video = 16

# read number of frames per video
frames_per_video = {}
frames_videos_file = "videos_frames_numfr_%d" % num_frames_per_video
if not os.path.exists(frames_videos_file):
    # read frames info
    for video in os.listdir(frames_directory):

        print("Reading video %s " % video)
        video_path = os.path.join(frames_directory,video)
        num_frames = len(os.listdir(video_path))

        if video in frames_per_video:
            print("%s already in the collection!" % video)
            exit(1)
        else:
            frame_list = [ i for i in range(num_frames)]
            shuffle(frame_list)
            frames_per_video[video] = frame_list[:16]

    with open(frames_videos_file,"wb") as ff:
        pickle.dump(frames_per_video, ff)
else:
    with open(frames_videos_file,"rb") as ff:
        frames_per_video = pickle.load(ff)


# read videos in phase
for phasefile in videos_in_phase_files:

    outfile = phasefile + ".singleframe_frames_" + str(num_frames_per_video)
    print("Writing " + outfile)
    videos_in_phase = []
    with open(phasefile, 'r') as fread:
        for phase_video in fread:
            video, label = phase_video.split()
            videos_in_phase.append((video, label))

    with open(outfile, 'w') as fwrite:
        for video, label in videos_in_phase:
            videoCategory, video = os.path.split(video)
            framenums = frames_per_video[video]
            for num in framenums:
                fwrite.write("%s/%s.%04d.jpg %s\n" % (video,video,1+num,label))


