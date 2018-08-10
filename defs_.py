from utils_ import error
import inspect
# constants, like C defines. Nesting indicates just convenient hierarchy.
class defs:
    # checks non full def names
    def check(arg, should_belong_to, do_boolean = False ):
        parts = arg.split(".")
        belongs_ok = False
        if not parts[0] == "defs":
            error("Invalid def : %s" % arg)
        else:
            curr_class = defs
            parts = parts[1:]
        for part in parts:
            if not belongs_ok:
                belongs_ok = should_belong_to == curr_class
            fields = inspect.getmembers(curr_class,  lambda a:not(inspect.isroutine(a)))
            fields = [v[0] for v in fields if not(v[0].startswith('__') or v[0].endswith('__'))]
            if not part in fields:
                if not do_boolean:
                    error('Parameter [%s] is not defined for [%s]' % (part, curr_class))
                else:
                    return (False, None)
            else:
                curr_class = getattr(curr_class, part)
        if not belongs_ok:
            if not do_boolean:
                error("Supplied parameter [%s] should be a child of def [%s]" % (arg, should_belong_to))
            else:
                return (False, None)
        if do_boolean:
            return (True, curr_class)
        else:
            return curr_class


    class representation:
        dcnn, fc, nop = "dcnn", "fc", "nop"
    class classifier:
        fc, lstm = "fc", "lstm"
    # run phase
    class phase:
        train, val ="train", "val"
    # input mode is framewise dataset vs videowise, each video having n frames
    class input_mode:
        video, image, vectors = "video", "image", "vectors"

    class net_input:
        visual, labels = "visual", "labels"

    # direct reading from disk or from packed tfrecord format
    class data_format:
        raw, tfrecord = "raw", "tfrecord"
    class rnn_visual_mode:
        state_bias, input_bias, input_concat = "state_bias", "input_bias", "input_concat"

    # sequence fusion methods
    class fusion_method:
        avg, last, concat, reshape, state, ibias, maximum = "avg", "last", "concat", "reshape", "state", "ibias", "maximum"

    # early/late fusion
    class fusion_type:
        early, late, none, main, aux  = "early", "late", "none", "main", "aux"

    # how the video's frames are structured
    class clipframe_mode:
        rand_frames, rand_clips, iterative = "rand_frames", "rand_clips", "iterative"

    # what to do if clip generation fails
    class generation_error:
        abort, compromise, report = "abort", "compromise", "report"

    class batch_item:
        default, clip = "default", "clip"

    class optim:
        sgd, rmsprop, adam = "sgd", "rmsprop", "adam"
        def adapts_lr(optimizer):
            return optimizer in [defs.optim.rmsprop, defs.optim.adam]
        def uses_momentum(optimizer):
            return optimizer not in [defs.optim.sgd]

    # learning rate decay parameters
    class decay:
        exp, staircase = "exp", "staircase"

    # periodicity
    class periodicity:
        interval, drops = "interval", "drops"

    class label_type:
        single, multiple = "single", "multiple"

    class caption_search:
        max = "max"

    class eval_type:
        coco = "coco"

    class names:
        global_step, latest_savefile = "global_step", "latest"

    class return_type:
        argmax_index, standard = "argmax_index", "standard"

    class imgproc:
        rand_mirror, rand_crop, center_crop, resize, raw_resize, sub_mean = \
                "rand_mirror", "rand_crop", "center_crop", "resize", "raw_resize", "sub_mean"
        def to_str(vec):
            res = []
            if defs.imgproc.rand_mirror in vec: res.append("rm")
            if defs.imgproc.rand_crop in vec: res.append("rc")
            if defs.imgproc.center_crop in vec: res.append("cc")
            if defs.imgproc.resize in vec: res.append("rs")
            if defs.imgproc.raw_resize in vec: res.append("rr")
            if defs.imgproc.sub_mean in vec: res.append("sm")
            return "-".join(res)
    train_idx, val_idx = 0, 1
    image, label = 0, 1
