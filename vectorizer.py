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
    No change; input is already in vector form
    """

    name = "nop"
    description = "Does not transform input"
    def __init__(self):
        Vectorizer.__init__(NOP.name, NOP.description)
    def build(self, io, settings):
        pass
    def forward(self, io):
        return io


class DCNN(Vectorizer, Trainable):
    """
    DCNN vectorizer
    """

    name = "dcnn"
    description = "Deep convolutional neural net"
    def __init__(self):
        Vectorizer.__init__(self, name = DCNN.name, description=DCNN.description)

    def build(self, io, settings):
        self.dcnn = dcnn()
        self.dcnn.create(io, None, settings.network.num_classes, settings.network.frame_encoding_layer, settings.network.load_weights)
        return self.dcnn.get_output()

class LSTM(Vectorizer, Trainable):
    name="lstm"
    description="Long short-term memory network"
    def __init__(self):
        Vectorizer.__init__(self, LSTM.name, LSTM.description)

    #def set_params(self):
    #    input_vec, input_state, len(input_vec), settings.network.lstm_params, output_dim,
    #                          sequence_length, nonzero_sequence, dropout_prob=0.0, omit_output_fc = False
    def build(self, io, settings):
        self.lstm = lstm()
        input_vec, input_dim, input_state, output_dim, seqlen, nonzero_seq, dropout, omit_fc = io
        output, state = self.lstm.forward_pass_sequence(input_vec, input_state, input_dim, settings.network.lstm_params,
                                        output_dim, seqlen, nonzero_seq, dropout, omit_fc)
        return output, state

