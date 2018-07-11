from defs_ import defs
from vectorizer import DCNN, NOP, LSTM, FC
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



    def build_pipeline(self, pipeline_name, settings):

        info("Building pipeline [%s]" % pipeline_name)

        # inits for multi-input layer
        inputs, cpvs, fpcs, dims = [], [], [], []
        pipeline =settings.pipelines[pipeline_name]
        # get settings
        pipeline_inputs = pipeline.input
        input_fusion = pipeline.input_fusion
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
        for i in  range(len(pipeline_inputs)):
            input_name = pipeline_inputs[i]
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
            dim = int(input.shape[-1])
            inputs.append(input)
            cpvs.append(cpv)
            fpcs.append(fpc)
            dims.append(dim)
            debug("Resolved pipeline input: [%s] shape,cpv,fpc,dim: [%s, %d, %d, %d]" % (input_name, str(input.shape),cpv, fpc, dim))

        # if len(inputs) > 1:
        #     return self.build_multi_pipeline(inputs, cpvs, fpcs, pipeline_name, settings)
        # else:

        # input fusion: for combination of multiple tensors - for anything more complex just declare a dedicated pipeline
        if input_fusion is not None:
            info("Applying input fusion with [%s]"% input_fusion)
            inputs, dims, fpcs, cpvs = apply_tensor_list_fusion(inputs, input_fusion, dims, fpcs, cpvs)
            inputs, dims, fpcs, cpvs = [inputs], [dims], [fpcs], [cpvs]

        # denote main pipeline inputs
        input = inputs[0]
        fpc = fpcs[0]
        output_fpc = fpc

        # vectorizer
        if repr == defs.representation.dcnn:
            repr_model = DCNN()
            self.tf_components.append(repr_model)
            feature_vectors = repr_model.build(input, (pipeline, settings.num_classes))
        elif repr == defs.representation.nop:
            repr_model = NOP()
            feature_vectors = repr_model.build(input, None)
        elif repr == defs.representation.fc:
            repr_model = FC()
            outdim = pipeline.fc_output_dim
            feature_vectors = repr_model.build(input, outdim)

        else:
            error("Undefined representation [%s]" % repr)

        dim = int(feature_vectors.shape[-1])
        feature_vectors = print_tensor(feature_vectors, "Vectorized output [%s]" % pipeline_name)

        # early fusion - aggregate before classification
        if fusion_type == defs.fusion_type.early and fpc > 1:
            feature_vectors = aggregate_clip_vectors(feature_vectors, dim, fpc, fusion_method=fusion_method)
            feature_vectors = print_tensor(feature_vectors, "Early fusion")
            output_fpc = 1
        elif fpc ==1 and fusion_type not in [defs.fusion_type.none, None]:
                info("Omitting specified fusion [%s][%s] due to singular fpc" % (fusion_type, fusion_method))

        if classif is None:
            self.pipeline_output_shapes[pipeline_name] = (feature_vectors.shape, cpv, output_fpc)
            return feature_vectors

        # classification
        if classif == defs.classifier.fc:
            if dim != num_classes:
                logits = convert_dim_fc(feature_vectors, num_classes)
            else:
                logits = feature_vectors
        elif classif == defs.classifier.lstm:
            if fpc == 1: error("The LSTM classifier requires an fpc greater than 1")
            if fusion_type is None:
                debug("Unset fusion type, setting it to explicitly to the def [%s]" % defs.fusion_type.none )
                fusion_type = defs.fusion_type.none
            if fusion_type != defs.fusion_type.none: error("The LSTM classifier should be used only with [%s] fusion, but it's [%s]" % (defs.fusion_type.none, fusion_type))
            classifier = LSTM()
            self.tf_components.append(classifier)
            if len(inputs) > 1:
                # consider the 2nd input as the state vector
                state_tensor, state_dim = inputs[1], dims[1]
                # replicate
                state_tensor = replicate_auxilliary_tensor(inputs, dims, cpvs, fpcs)
            else:
                state_tensor = None
            io_params = (feature_vectors, dim, state_tensor, num_classes, fpc, None, settings.get_dropout(), False)
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
        for pname in settings.pipeline_names:
            pipeline_output = self.build_pipeline(pname, settings)
            self.pipeline_output[pname] = pipeline_output
        # get last defined element for the output
        self.logits = self.pipeline_output[settings.pipeline_names[-1]]

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
