from utils_ import  *


def apply_temporal_pooling(input_tensor, vector_dimension, temporal_dimension, pooling_type=defs.pooling.reshape, name="lstm_temporal_pooling"):
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
    elif pooling_type == defs.pooling.reshape:
        output = tf.reshape(input_tensor,[-1, vector_dimension])
    else:
        error("Undefined frame pooling type : %d" % pooling_type)
    return output


def convert_dim_fc(input_tensor, output_dim, name="fc_convert", reuse = False):
    """
    Make and apply a fully-connected layer to map the input_dim to the output_dim, if needed
    """
    # input_tensor = print_tensor(input_tensor, "Input to fc-convert with name %s : " % name)
    input_dim = int(input_tensor.shape[1])
    if input_dim  == output_dim:
        return input_tensor
    if not reuse:
        # layer initializations
        fc_out__init = tf.truncated_normal((input_dim, output_dim), stddev=0.05, name=name + "_w_init")
        fc_out_b_init = tf.constant(0.1, shape=(output_dim,), name=name + "_b_init")

        # create the layers
        fc_out_w = tf.get_variable(initializer=fc_out__init, name=name + "_w")
        fc_out_b = tf.get_variable(initializer=fc_out_b_init, name=name + "_b")

        # fc_out_w = tf.Variable(fc_out__init, name=name + "_w")
        # fc_out_b = tf.Variable(fc_out_b_init, name=name + "_b")
    else:
        fc_out_w = tf.get_variable(name + "_w")
        fc_out_b = tf.get_variable(name + "_b")
    output = tf.nn.xw_plus_b(input_tensor, fc_out_w, fc_out_b, name=name)
    # output = print_tensor(output, "Output from fc-convert with name %s" % name)

    return output


