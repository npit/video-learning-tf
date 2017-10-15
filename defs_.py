from utils_ import error
# constants, like C defines. Nesting indicates just convenient hierarchy.
class defs:

    # run phase
    class phase:
        train, val ="train", "val"

    # input mode is framewise dataset vs videowise, each video having n frames
    class input_mode:
        video, image = "video", "image"
        def get_from_workflow(arg):
            if defs.workflows.is_image(arg):
                return defs.input_mode.image
            elif defs.workflows.is_video(arg):
                return defs.input_mode.video
            else:
                error("No input mode discernible from workflow %s" % arg)
                return None

    # direct reading from disk or from packed tfrecord format
    class data_format:
        raw, tfrecord = "raw", "tfrecord"
    class rnn_visual_mode:
        state_bias, input_bias, input_concat = "state_bias", "input_bias", "input_concat"
    # run type indicates usage of lstm or singleframe dcnn
    class workflows:
        class acrec:
            singleframe, lstm = "acrec_singleframe", "acrec_lstm"
            def is_workflow(arg):
                return arg == defs.workflows.acrec.singleframe or \
                       arg == defs.workflows.acrec.lstm
        class imgdesc:
            statebias, inputstep, inputbias = "imgdesc_statebias", "imgdesc_inputstep", "imgdesc_inputbias"
            def is_workflow(arg):
                return arg == defs.workflows.imgdesc.statebias or \
                       arg == defs.workflows.imgdesc.inputstep or \
                       arg == defs.workflows.imgdesc.inputbias
        class videodesc:
            pooled, encdec = "videodesc_pooled", "videodesc_encdec"
            def is_workflow(arg):
                return arg == defs.workflows.videodesc.pooled or \
                       arg == defs.workflows.videodesc.encdec
        def is_description(arg):
            return defs.workflows.imgdesc.is_workflow(arg) or \
                    defs.workflows.videodesc.is_workflow(arg)
        def is_video(arg):
            return defs.workflows.acrec.singleframe == arg or \
                   defs.workflows.acrec.lstm== arg or \
                   defs.workflows.videodesc.encdec == arg or \
                   defs.workflows.videodesc.pooled == arg
        def is_image(arg):
            return defs.workflows.imgdesc.statebias == arg or \
                   defs.workflows.imgdesc.inputstep == arg or \
                   defs.workflows.imgdesc.inputbias == arg
                #lstm, singleframe, imgdesc, videodesc = "lstm","singleframe", "imgdesc", "videodesc"

    # video pooling methods
    class pooling:
        avg, last, reshape = "avg", "last", "reshape"

    # how the video's frames are structured
    class clipframe_mode:
        rand_frames, rand_clips, iterative = "rand_frames", "rand_clips", "iterative"

    class batch_item:
        default, clip = "default", "clip"

    class optim:
        sgd, adam = "sgd", "adam"

    # learning rate decay parameters
    class decay:
        # granularity level
        class granularity:
            exp, staircase = "exp", "staircase"
        # drop at intervals or a total number of times
        class scheme:
            interval, total = "interval", "total"

    class label_type:
        single, multiple = "single", "multiple"

    class caption_search:
        max = "max"

    class eval_type:
        coco = "coco"
    class variables:
        global_step = "global_step"
    class return_type:
        argmax_index, standard = "argmax_index", "standard"
    train_idx, val_idx = 0, 1
    image, label = 0, 1
