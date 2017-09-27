from utils_ import  *


def apply_temporal_pooling(input_tensor, vector_dimension, temporal_dimension, pooling_type):
    '''
    Apply pooling over the temporal (column) dimension of the input tensor
    :param self:
    :param input_tensor:
    :param vector_dimension:
    :param temporal_dimension:
    :param pooling_type:
    :return:
    '''
    if pooling_type == defs.pooling.last:
        # keep only the response at the last time step
        output = tf.slice(input_tensor, [0, temporal_dimension - 1, 0], [-1, 1, vector_dimension], name="lstm_output_reshape")
        debug("LSTM last timestep output : %s" % str(output.shape))
        # squeeze empty dimension to get vector
        output = tf.squeeze(output, axis=1, name="lstm_output_squeeze")
        debug("LSTM squeezed output : %s" % str(output.shape))

    elif pooling_type == defs.pooling.avg:
        # average per-timestep results
        output = tf.reduce_mean(input_tensor, axis=1)
        debug("LSTM time-averaged output : %s" % str(output.shape))
    else:
        error("Undefined frame pooling type : %d" % pooling_type)
    return output

def make_fc(input_tensor, input_dim, output_dim, name="fc_out" ):
    '''
    Make a fully-connected layer
    '''
    # layer initializations
    fc_out__init = tf.truncated_normal((input_dim, output_dim), stddev=0.05, name=name + "_w_init")
    fc_out_b_init = tf.constant(0.1, shape=(output_dim,), name=name + "_b_init")

    # create the layers
    fc_out_w = tf.Variable(fc_out__init, name=name + "_w")
    fc_out_b = tf.Variable(fc_out_b_init, name=name + "_b")
    output = tf.nn.xw_plus_b(input_tensor, fc_out_w, fc_out_b, name=name)

    return output