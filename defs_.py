from utils_ import error
import inspect
# constants, like C defines. Nesting indicates just convenient hierarchy.
class defs:
    # checks non full def names
    def check(arg, should_belong_to):
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
                error('Parameter [%s] is not defined for [%s]' % (part, curr_class))
            else:
                curr_class = getattr(curr_class, part)
        if not belongs_ok:
            error("Supplied parameter [%s] should be a child of def [%s]" % (arg, should_belong_to))
        return curr_class

    # checks full def names
    def check_full(self,arg):
        parts = arg.split(".")
        if not parts[0] == "defs":
            error("Invalid def : %s" % arg)
        curr_class = defs
        for part in parts[1:]:
            curr_class = defs.check_part(part, curr_class)
        return curr_class


    class representation:
        dcnn, nop = "dcnn", "nop"
    class classifier:
        fc, lstm = "fc", "lstm"

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
            elif defs.check("defs.workflows.multi."+arg, defs.workflows.multi):
                return defs.input_mode.video
            else:
                error("No input mode discernible from workflow %s" % arg)
                return None

    class net_input:
        visual, labels = "visual", "labels"

    class dataset_tag:
        main, aux = "main", "aux"

    # direct reading from disk or from packed tfrecord format
    class data_format:
        raw, tfrecord = "raw", "tfrecord"
    class rnn_visual_mode:
        state_bias, input_bias, input_concat = "state_bias", "input_bias", "input_concat"
    # run type indicates usage of lstm or singleframe dcnn
    class workflows:
        class multi:
            fc, singleframe, lstm, lstm_sbias, lstm_ibias, lstm_conc = "fc", "singleframe", "lstm", "lstm_sbias", "lstm_ibias", "lstm_conc"
        class acrec:
            singleframe, lstm, audio, multi = "acrec_singleframe", "acrec_lstm", "acrec_audio", "multi"
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
            fused, encdec = "videodesc_fused", "videodesc_encdec"
            def is_workflow(arg):
                return arg == defs.workflows.videodesc.fused or \
                       arg == defs.workflows.videodesc.encdec
        def is_description(arg):
            return defs.workflows.imgdesc.is_workflow(arg) or \
                    defs.workflows.videodesc.is_workflow(arg)
        def is_video(arg):
            return defs.workflows.acrec.singleframe == arg or \
                   defs.workflows.acrec.lstm == arg or \
                   defs.workflows.videodesc.encdec == arg or \
                   defs.workflows.videodesc.fused == arg or \
                   defs.workflows.acrec.audio == arg or \
                   defs.workflows.acrec.multi == arg
        def is_image(arg):
            return defs.workflows.imgdesc.statebias == arg or \
                   defs.workflows.imgdesc.inputstep == arg or \
                   defs.workflows.imgdesc.inputbias == arg
                #lstm, singleframe, imgdesc, videodesc = "lstm","singleframe", "imgdesc", "videodesc"

    # sequence fusion methods
    class fusion_method:
        avg, last, concat, reshape, state = "avg", "last", "concat", "reshape", "state"

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
        sgd, adam = "sgd", "adam"

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
