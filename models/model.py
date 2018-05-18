from defs_ import defs
from vectorizer import DCNN, NOP, LSTM
from utils_ import error, info, debug, drop_tensor_name_index, print_tensor
from tf_util import *
import tensorflow as tf

class Model:
    """
    A model, consisting of a representation and a classifier
    """
    tf_components = []
    required_input = []
    pipeline_output = {}
    pipeline_output_shapes = {}

    def handle_input(self, settings):
        # if not multi, input is single data and labels
        if settings.workflow != defs.workflows.multi:
            self.input = tf.placeholder(tf.float32, (None,) + settings.network.image_shape, name='input_frames')
            #self.labels = tf.placeholder(tf.int32, (None, settings.network.num_classes), name='input_labels')
            self.required_input.append((self.input, defs.net_input.visual, defs.dataset_tag.main))
            #self.input_config.append((self.labels, defs.net_input.labels, defs.dataset_tag.main))
        else:
            # else, needs double that
            pass


    def get_cpv_information(self, pipeline, settings):
        cpvs = []
        input_names = pipeline.input
        for inp in input_names:
            if inp in settings.pipelines:
                pass
        return cpvs


    def build_multi_pipeline(self, inputs, cpvs, fpcs, pipeline_name, settings):
        # guaranteed not to have a representation operator
        # this pipeline's sole purpose is to fuse multiple inputs to one.
        # all other scenarios should be handled by the regular variant
        # get settings
        pipeline = settings.pipelines[pipeline_name]
        input_names = pipeline.input
        classif = pipeline.classifier
        num_classes = settings.num_classes
        if pipeline.frame_fusion:
            fusion_type, fusion_method = pipeline.frame_fusion
        else:
            fusion_type, fusion_method = None, None
        cpv1, cpv2 = cpvs

        if fusion_type == defs.fusion_type.early:
            input = aggregate_tensor_list(inputs, fusion_method)
            if fusion_method == defs.fusion_method.avg:
                fpc = fpcs[0]
            elif fusion_method == defs.fusion_method.concat:
                fpc = sum(fpcs)

        if classif == defs.classifier.lstm and len(inputs) == 2:
            input1, input2 = inputs
            dim1, dim2 = int(input1.shape[-1]), int(input2.shape[-1])
            fpc1, fpc2 = fpcs
            lstm_params = pipeline.lstm_params
            # exactly one dataset needs to be frame-fused
            if fpc1 == 1: error("The LSTM-dual classifier requires a main fpc greater than 1")
            if fpc2 != 1: error("The LSTM-dual classifier requires an auxilliary fpc equal to 1, found [%d] instead." % fpc2)
            classifier = LSTM()
            self.tf_components.append(classifier)
            combo_type = lstm_params[3]
            if combo_type == defs.combo.sbias:
                # tile the aux, if necessary
                tile_num = int(cpv1/cpv2)
                if tile_num > 1:
                    input2 = tf.reshape(input2, [1, -1])
                    input2 = tf.tile(input2, [tile_num, 1])
                    input2 = tf.reshape(input2, [-1, dim2])
                # supply inputs
                io_params = (input1, dim1, input2, num_classes, fpc1, None, settings.get_dropout(), False)
                lstm_output, lstm_state = classifier.build(io_params, lstm_params)
            elif combo_type == defs.combo.conc:
                # concat to input
                tile_num = int(cpv1/cpv2)
                if tile_num > 1:
                    input2 = tf.reshape(input2, [1, -1])
                    input2 = tf.tile(input2, [tile_num, 1])
                    input2 = tf.reshape(input2, [-1, dim2])
                input = vec_seq_concat(input1, input2, fpc1)
                io_params = (input, dim1 + dim2, None, num_classes, fpc1, None, settings.get_dropout(), False)
                lstm_output, lstm_state = classifier.build(io_params, lstm_params)
            elif combo_type == defs.combo.ibias:
                if dim2 != dim1:
                    warning("Transforming with an fc the dim. of [%s] data to match the [%s]: [%d] -> [%d]" % (input_names[1],input_names[0], dim2, dim1))
                    input2 = convert_dim_fc(input2, dim1, name="lstm_multi_fc_convert")
                    dim2 = dim1
                # reshape seq vector to numclips x fpc x dim
                reshaped_seq_dset= tf.reshape(input1, [-1, fpc1, dim1])
                reshaped_seq_dset = print_tensor(reshaped_seq_dset,"reshaped seq")
                # duplicate bias vector to fpc
                tile_num = int(cpv1/cpv2)
                input2_tiled = tf.reshape(input2, [1, -1])
                input2_tiled = tf.tile(input2_tiled, [tile_num, 1])
                input2_tiled = print_tensor(input2_tiled,"tiled bias")
                # reshape the bias dataset to batch_size x fpc=1 x dim
                reshaped_bias_dset = tf.reshape(input2_tiled, [-1, 1, dim2])
                reshaped_bias_dset = print_tensor(reshaped_bias_dset,"reshaped bias")
                # insert the fused as the first item in the seq - may need tf.expand on the fused
                input_biased_seq = tf.concat([reshaped_bias_dset, reshaped_seq_dset], axis=1)
                # increase the seq len to account for the input bias extra timestep
                augmented_fpc = fpc1 + 1
                info("Input bias augmented fpc: %d + 1 = %d" % (fpc1, augmented_fpc))
                # restore to batchsize*seqlen x embedding_dim
                input_biased_seq = tf.reshape(input_biased_seq ,[-1, dim1])

                # supply inputs
                io_params = (input_biased_seq, dim1, None, num_classes, augmented_fpc, None, settings.get_dropout(), False)
                lstm_output, lstm_state = classifier.build(io_params, lstm_params)
            else:
                error("Undefined combo type: [%s]" % combo_type)

            if lstm_params[2] == defs.fusion_method.state:
                logits = lstm_state[-1].h
            else:
                logits = lstm_output

            if int(logits.shape[1]) != num_classes:
                logits = convert_dim_fc(logits, num_classes)
            self.pipeline_output[pipeline_name] = logits
            self.pipeline_output_shapes[pipeline_name] = (logits.shape, cpv1, 1)
        return logits



    def build_pipeline(self, pipeline_name, settings):

        info("Building pipeline [%s]" % pipeline_name)

        # inits for multi-input layer
        inputs, cpvs, fpcs = [], [], []

        pipeline =settings.pipelines[pipeline_name]
        # get settings
        pipeline_inputs = pipeline.input
        repr = pipeline.representation
        classif = pipeline.classifier
        num_classes = settings.num_classes
        if pipeline.frame_fusion:
            fusion_type, fusion_method = pipeline.frame_fusion
        else:
            fusion_type, fusion_method = None, None

        # sanity checks
        if classif is None and fusion_type == defs.fusion_type.late:
            error("Specified late fusion with no classifier selected")

        # define input
        # combination-only pipeline
        for i in  range(len(pipeline_inputs)):
            input_name = pipeline_inputs[i]
            debug("Resolving pipeline input: [%s]" % input_name)
            if input_name in settings.pipelines:
                # get input from existing pipeline
                input = self.pipeline_output[input_name]
                _, cpv, fpc = self.pipeline_output_shapes[input_name]
            else:
                shp = pipeline.input_shape[i]
                if shp is None:
                    shp = settings.feeder.get_dataset_by_tag(input_name)[0].get_image_shape()
                # shp has to be the same as the image shape in the dataset configuration
                input = tf.placeholder(tf.float32, (None,) + shp, name='%s_%s_input' % (pipeline_name, input_name))
                self.required_input.append((input, defs.net_input.visual, input_name))
                cpv = settings.feeder.get_dataset_by_tag(input_name)[0].clips_per_video
                if not all(cpv[0] == c for c in cpv):
                    warning("Non equal clips per item")
                cpv = cpv[0]
                fpc = settings.feeder.get_dataset_by_tag(input_name)[0].num_frames_per_clip
            inputs.append(input)
            cpvs.append(cpv)
            fpcs.append(fpc)

        if len(inputs) > 1:
            return self.build_multi_pipeline(inputs, cpvs, fpcs, pipeline_name, settings)
        else:
            input = inputs[-1]
            fpc = fpcs[-1]
            output_fpc = fpc

        # vectorizer
        if repr == defs.representation.dcnn:
            repr_model = DCNN()
            self.tf_components.append(repr_model)
            feature_vectors = repr_model.build(input, (pipeline, settings.num_classes))
        elif repr == defs.representation.nop:
            repr_model = NOP()
            feature_vectors = repr_model.build(input, None)
        else:
            error("Undefined representation [%s]" % repr)

        dim = int(feature_vectors.shape[-1])
        feature_vectors = print_tensor(feature_vectors, "Vectorized output [%s]" % pipeline_name)

        # early fusion - aggregate before classification
        if fusion_type == defs.fusion_type.early and fpc > 1:
            feature_vectors = aggregate_clip_vectors(feature_vectors, dim, fpc, fusion_method=fusion_method)
            feature_vectors = print_tensor(feature_vectors, "Early fusion")
            output_fpc = 1
        elif fpc ==1 and fusion_type != defs.fusion_type.none:
                info("Omitting specified fusion [%s][%s] due to singular fpc" % (fusion_type, fusion_method))

        if classif is None:
            self.pipeline_output_shapes[pipeline_name] = (feature_vectors.shape, cpv, output_fpc)
            return feature_vectors

        # classification
        if classif == defs.classifier.fc:
            if dim != num_classes:
                logits = convert_dim_fc(feature_vectors, num_classes)
            else:
                pass
        elif classif == defs.classifier.lstm:
            if fpc == 1: error("The LSTM classifier requires an fpc greater than 1")
            if fusion_type != defs.fusion_type.none: error("The LSTM classifier should be used only with [%s] fusion" % defs.fusion_type.none)
            classifier = LSTM()
            self.tf_components.append(classifier)
            io_params = (feature_vectors, dim, None, num_classes, fpc, None, settings.get_dropout(), False)
            lstm_output, lstm_state = classifier.build(io_params, pipeline.lstm_params)
            if pipeline.lstm_params[-1] == defs.fusion_method.state:
                logits = lstm_state[-1].h
            else:
                logits = lstm_output

            if int(logits.shape[1]) != num_classes:
                logits = convert_dim_fc(logits, num_classes)
        else:
            error("Undefined classifier [%s]" % repr)

        logits = print_tensor(logits, "Post-classification logits")
        # late fusion - aggregate after classification
        if fusion_type == defs.fusion_type.late and fpc > 1:
            logits = aggregate_clip_vectors(logits, num_classes, fpc, fusion_method=fusion_method)
            logits = print_tensor(logits, "Late fusion")

        self.pipeline_output_shapes[pipeline_name] = (logits.shape, cpv, 1)
        logits = print_tensor(logits, "Final logits")
        return logits




    def __init__(self, settings):
        pnames = sorted(settings.pipelines, key = lambda x : settings.pipelines[x].idx)
        for pname in pnames:
            pipeline_output = self.build_pipeline(pname, settings)
            self.pipeline_output[pname] = pipeline_output
        # get last defined element for the output
        last_name = max(settings.pipelines, key=lambda x : settings.pipelines[x].idx)
        self.logits = self.pipeline_output[last_name]

    def __init2__(self, settings):


        """
        each component is a list

        if len > 1, gotta fuse somehow

        temporal point of fusion depends on early or late

        mutli workflow gr8ly simplified
        """

        # get settings
        repr = settings.network.representation
        classif = settings.network.classifier
        num_classes = settings.network.num_classes
        fpc = settings.feeder.get_dataset_by_tag(defs.dataset_tag.main)[0].num_frames_per_clip
        fusion_type = settings.network.frame_fusion_type
        fusion_method = settings.network.frame_fusion_method


        if fpc ==1 and fusion_type != defs.fusion_type.none:
            info("Omitting specified fusion [%s][%s] due to singular fpc" % (fusion_type, fusion_method))

        # print model
        components = [repr]
        if components[-1] == DCNN.name: components[-1] += "-" + settings.network.frame_encoding_layer
        if fusion_type == defs.fusion_type.early: components.append(fusion_method)
        components.append(classif)
        if components[-1] == LSTM.name: components[-1] += "-" + "-".join(map(str,settings.network.lstm_params))
        if fusion_type == defs.fusion_type.late: components.append(fusion_method)
        info("Model:[%s]" % ",".join(components))

        # define input
        # ------------
        self.handle_input(settings)

        # vectorizer
        self.repr_model = None
        self.feature_vectors = None

        if repr == defs.representation.dcnn:
            self.repr_model = DCNN()
            self.tf_components.append(self.repr_model)
        elif repr == defs.representation.nop:
            self.repr_model = NOP()
        else:
            error("Undefined representation [%s]" % repr)

        self.feature_vectors = self.repr_model.build(self.input, settings)
        self.feature_dim = int(self.feature_vectors.shape[-1])
        self.feature_vectors = print_tensor(self.feature_vectors, "Vectorized output")

        # early fusion - aggregate before classification
        if fusion_type == defs.fusion_type.early and fpc > 1:
            self.feature_vectors = aggregate_clip_vectors(self.feature_vectors, self.feature_dim, fpc, fusion_method=fusion_method)
            self.feature_vectors = print_tensor(self.feature_vectors, "Early fusion")

        # classification
        self.logits = None
        if classif == defs.classifier.fc:
            if self.feature_dim != num_classes:
                self.classifier = None
                self.logits = convert_dim_fc(self.feature_vectors,num_classes)
            else:
                self.logits = self.feature_vectors
        elif classif == defs.classifier.lstm:
            if fpc == 1: error("The LSTM classifier requires an fpc greater than 1")
            if fusion_type != defs.fusion_type.none: error("The LSTM classifier should be used only with [%s] fusion" % defs.fusion_type.none)
            self.classifier = LSTM()
            self.tf_components.append(self.classifier)
            io_params = (self.feature_vectors, self.feature_dim, None, num_classes, fpc, None, settings.get_dropout(), False)
            output, state = self.classifier.build(io_params, settings)
            if settings.network.lstm_params[-1] == defs.fusion_method.state:
                self.logits = state[-1].h
            else:
                self.logits = output

            if int(self.logits.shape[1]) != num_classes:
                self.logits = convert_dim_fc(self.logits, num_classes)
        else:
            error("Undefined classifier [%s]" % repr)

        self.logits = print_tensor(self.logits, "Post-classification logits")
        # late fusion - aggregate after classification
        if fusion_type == defs.fusion_type.late and fpc > 1:
            self.logits = aggregate_clip_vectors(self.logits, num_classes, fpc, fusion_method=fusion_method)
            self.logits = print_tensor(self.logits, "Late fusion")

        self.logits = print_tensor(self.logits, "Final logits")


    def get_output(self):
        return self.logits

    def get_ignorable_variable_names(self):
        ignorables = []

        for component in self.tf_components:
            if component.ignorable_variable_names:
                ignorables.extend(component.ignorable_variable_names)
        if ignorables:
            info("Getting raw ignorables: %s" % str(ignorables))
            ignorables = [drop_tensor_name_index(s) for s in ignorables]
        return list(set(ignorables))


