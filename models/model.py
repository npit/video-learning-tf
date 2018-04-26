from defs_ import defs
from vectorizer import DCNN, NOP, LSTM
from utils_ import error, info, debug, drop_tensor_name_index, print_tensor
from tf_util import aggregate_clip_vectors, convert_dim_fc
import tensorflow as tf

class Model:
    """
    A model, consisting of a representation and a classifier
    """
    tf_components = []
    required_input = []

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

    def __init__(self, settings):
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
            assert fusion_type == defs.fusion_type.none, "The LSTM classifier should be used only with [%s] fusion" % defs.fusion_type.none
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


