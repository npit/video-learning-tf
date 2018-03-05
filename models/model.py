from defs_ import defs
from vectorizer import DCNN, NOP, LSTM
from utils_ import error
from tf_util import apply_temporal_fusion, convert_dim_fc
import tensorflow as tf

class Model:
    """
    A model, consisting of a representation and a classifier
    """
    def __init__(self, settings):
        # get run parameters
        fpc = settings.get_dataset_by_tag(defs.dataset_tag.main)[0].num_frames_per_clip
        fusion_type = settings.network.frame_fusion_type
        fusion_method = settings.network.frame_fusion_method

        # define input
        self.input = tf.placeholder(tf.float32, (None,) + settings.network.image_shape, name='input_frames')

        # representation
        repr = settings.network.representation
        self.repr_model = None
        self.feature_vectors = None

        if repr == defs.representation.dcnn:
            self.repr_model = DCNN()
            self.feature_vectors = self.repr_model.build(self.input, settings)
        elif repr == defs.representation.nop:
            self.repr_model = NOP()
            self.feature_vectors = self.repr_model.build(self.input, settings)
        else:
            error("Undefined representation [%s]" % repr)

        self.feature_dim = int(self.feature_vectors.shape[-1])

        # early fusion
        if fusion_type == defs.fusion_type.early:
            self.feature_vectors = apply_temporal_fusion(self.feature_vectors, self.feature_dim, fpc)

        # classification
        classif = settings.network.classifier
        num_classes = settings.network.num_classes
        self.logits = None
        if classif == defs.classifier.fc:
            if self.feature_dim != num_classes:
                self.logits = convert_dim_fc(self.feature_vectors,num_classes)
            else:
                self.logits = self.feature_vectors
        elif classif == defs.classifier.lstm:
            classif_model = LSTM()
            io_params = (self.feature_vectors, None, num_classes, fpc, None, settings.get_dropout(), False)
            self.logits, _ = classif_model.build(io_params, settings)
        else:
            error("Undefined classifier [%s]" % repr)

        # late fusion
        if fusion_type == defs.fusion_type.late:
            self.logits = apply_temporal_fusion(self.logits, len(self.logits), fpc)


        if settings.phase == defs.phase.val:
            return

        # training
        self.labels = tf.placeholder(tf.int32, [None, settings.network.num_classes], name="input_labels")


