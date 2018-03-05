from utils_ import Trainable
from models.alexnet.alexnet import dcnn
from models.lstm.lstm import lstm

class Vectorizer:
    """
    Class responsible for transforming raw input data to a feature vector
    """
    def __init__(self, name, description):
        self.name = name
        self.description = description
    def __str__(self):
        return "%s:%s" % (self.name, self.description)
    def build(self, io, settings):
        pass


class NOP(Vectorizer):
    """
    No change; input is already vectors
    """
    def __init__(self, name):
        Vectorizer.__init__(name, "Does not transform its input")
    def build(self, io, settings):
        pass
    def forward(self, io):
        return io


class DCNN(Vectorizer, Trainable):
    """
    DCNN vectorizer
    """

    def __init__(self):
        Vectorizer.__init__(self, name = "dcnn", description="Deep convolutional neural net")

    def build(self, io, settings):
        self.dcnn = dcnn()
        self.dcnn.create(io, None, settings.network.num_classes, settings.network.frame_encoding_layer, settings.network.load_weights)
        return self.dcnn.get_output()

class LSTM(Vectorizer, Trainable):
    def __init__(self):
        Vectorizer.__init__("lstm", "Long Short-Term Memory net")

    #def set_params(self):
    #    input_vec, input_state, len(input_vec), settings.network.lstm_params, output_dim,
    #                          sequence_length, nonzero_sequence, dropout_prob=0.0, omit_output_fc = False
    def build(self, io, settings):
        self.lstm = lstm()
        input_vec, input_state, output_dim, seqlen, nonzero_seq, dropout, omit_fc = io
        self.lstm.forward_pass_sequence(input_vec, input_state, len(input_vec), settings.network.lstm_params, output_dim,
                              seqlen, nonzero_seq, dropout, omit_fc)

