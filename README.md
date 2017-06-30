# Image / Video classification and description using tensorflow
This is a python3 / tensorflow implementation of the architectures described in [Donahue et al. 2015](https://arxiv.org/abs/1411.4389) for image & video classification and description.
## Overview
- Use the `run_task.py` script to begin the execution of a workflow.
- Use the `config.ini` files to provide execution or serialization parameters to the associated scripts. Parameter values should be python3 code.
- You can see non-primitive supported parameters and options in `utils_.defs`
## Data preprocessing : `serialize.py`
Tensorflow recommended serialization format is the [TFRecord ](https://www.tensorflow.org/programmers_guide/reading_data). You can serialize a list of images or videos by using the `serialize.py` script.

To serialize a list of items (images or videos), provide a list with file paths to the `input_files` ini variable, followed by their label index(es). You can find examples in `examples/test_run` folder per data type. Use the `path_prepend_folder` variable to complete relative paths in the input files.
Each file path should contain an image name or a folder containing video frames. The former's encoding should be specified at the `frame_format` ini variable. If the entries in the input files do not match the given `frame_format`, it is assumed they constitute video folders. Frames in each video folder should be named as `N.fmt, N={1,2,...}` and `fmt` the image encoding.

Resources and workflow control:
- `num_threads`: number of threads to use for serialization
- `num_items_per_thread` : max number of videos or images assigned per thread
- `do_shuffle`, `do_serialze`: Shuffle the paths within each file, do the serialization.

You can control how and which frames are selected from all available frames within a video folder. Clips refer to an ordered collection of sequential frames. 

The major clip and / or frame generation modes for each video (`defs.clipframe_mode`) are:
- `rand_frames`: Select (unique) frames randomly
- `rand_clips`: Select (non-unique) clips randomly
- `iterative`: Select clips starting from the first frame, leaving a fixed frame offset between clips

Variables for video generation are :

- `clip_offset_or_num`: Either the number of clips for `rand_clips` generation, or the frame offset between clips, for `iterative`
- `num_frames_per_clip`: Either the number of frames within each clip.
- `raw_image_shape`: Image resize dimensions 
- `clipframe_mode`: The clip / frame generation mode
- `frame_format`: The image format 

The generated files for an input of `data.train` include
- `data.train.shuffled`: the output shuffled paths, if `do_shuffle` is enabled
- `data.train.tfrecord`: the tfrecord serialization, if `do_serialize` is enabled
- `data.train.tfrecord.size`: metadata containing the number of items, the number of frames per video and the number of clips of video, for a `.tfrecord` file.

## Workflows
Available workflows are defined in `defs.workflows` and explained below.
### Activity recognition 
The activity recognition workflows classify videos to a predifined number of classes. It can be instantiated by the following two workflows.
#### Single-frame 
The single-frame workflow uses a Alexnet DCNN to classify each video frame individually. Video-level predictions are produced by pooling the predicted label of each video frame using an aggregation method defined in `defs.pooling`.
#### LSTM
The lstm workflow uses an LSTM to classify a video taking into account the temporal dynamics across the video frames. Per-frame predictions are pooled similarly to the single-frame case.
### Image description
The image description workflow produces captions for a given input image. 
