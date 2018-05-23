from utils_ import  *
from defs_ import *

def apply_temporal_fusion(input_tensor, vector_dimension, temporal_dimension, fusion_method=defs.fusion_method.reshape, name="temporal_fusion", lstm_encoder = None ):
    '''
    Apply fusion over the temporal (column) dimension of the input tensor
    :param self:
    :param input_tensor:
    :param vector_dimension:
    :param temporal_dimension:
    :param fusion_method:
    :return:
    '''
    if fusion_method == defs.fusion_method.state:
        return None

    if fusion_method == defs.fusion_method.last:
        # keep only the response at the last time step
        output = tf.slice(input_tensor, [0, temporal_dimension - 1, 0], [-1, 1, vector_dimension], name="lstm_output_reshape")
        #debug("LSTM last timestep output : %s" % str(output.shape))
        # squeeze empty dimension to get vector
        output = tf.squeeze(output, axis=1, name="lstm_output_squeeze")
        debug("Agreggated last-squeezed output : %s" % str(output.shape))

    elif fusion_method == defs.fusion_method.avg:
        # average per-timestep results
        output = tf.reduce_mean(input_tensor, axis=1)
        debug("Aggregated time-averaged output : %s" % str(output.shape))
    elif fusion_method == defs.fusion_method.reshape:
        output = tf.reshape(input_tensor,[-1, vector_dimension])
    elif fusion_method == defs.fusion_method.lstm:
        if lstm_encoder == None:
            error("Did not provide an lstm encoder for fusion.")
        # fuse via lstm encoding; simplest lstm setting: 1 layer, statedim = inputdim, outputdim = 8
        # setting output dim to a dummy dim of 8, since we'll get the state
        lstm_params = lstm_encoder.params
        _, output = lstm_encoder.forward_pass_sequence(input_tensor, None, vector_dimension, lstm_params, 8, temporal_dimension, None, 0.5, omit_output_fc=True)
        # get the state h vector
        output = output[0].h
        info("Aggregated lstm output [%s]" % str(output.shape))
    else:
        error("Undefined frame fusion type : %s" % str(fusion_method))
    return output

def convert_dim_fc(input_tensor, output_dim, name="fc_convert", reuse = False):
    """
    Make and apply a fully-connected layer to map the input_dim to the output_dim, if needed
    """
    # input_tensor = print_tensor(input_tensor, "Input to fc-convert with name %s : " % name)
    input_dim = int(input_tensor.shape[-1])
    info("Converting dim %d to %d with fc layer" % (input_dim, output_dim))
    if input_dim  == output_dim:
        return input_tensor
    input_shape = input_tensor.shape
    if not reuse:
        # layer initializations
        fc_out_init = tf.truncated_normal((input_dim, output_dim), stddev=0.05, name=name + "_w_init")
        fc_out_b_init = tf.constant(0.1, shape=(output_dim,), name=name + "_b_init")

        # create the layers
        fc_out_w = tf.get_variable(initializer=fc_out_init, name=name + "_w")
        fc_out_b = tf.get_variable(initializer=fc_out_b_init, name=name + "_b")

        # fc_out_w = tf.Variable(fc_out__init, name=name + "_w")
        # fc_out_b = tf.Variable(fc_out_b_init, name=name + "_b")
    else:
        fc_out_w = tf.get_variable(name + "_w")
        fc_out_b = tf.get_variable(name + "_b")
    output = tf.nn.xw_plus_b(input_tensor, fc_out_w, fc_out_b, name=name)
    # output = print_tensor(output, "Output from fc-convert with name %s" % name)

    debug("F [%s]: %s * %s + %s = %s" % (name,str(input_shape), str(fc_out_w.shape), str(fc_out_b.shape), str(output.shape)))
    return output

def vectorize(input_tensor, depth_dim):
    return tf.reshape(input_tensor, [ -1, depth_dim])

# dcnn helpers
def make_fusion(input_tensor, window, strides, padding, name):
    """
    Pooling definition helper function
    :param input_tensor:
    :param window:
    :param strides:
    :param padding:
    :param name:
    :return:
    """
    value = tf.nn.max_pool(input_tensor, ksize = window, strides= strides, padding = padding, name=name)
    debug("P [%s]: %s -> %s" % (name,str(input_tensor.shape), str(value.shape)))
    return value

def make_conv(input_tensor, kernel_params, strides, scopename, init_w=(0.0, 0.1), padding="SAME" ):
    """
    Convolution definition helper function
    :param input_tensor:
    :param kernel_params:
    :param strides:
    :param scopename:
    :param init_w:
    :param padding:
    :return:
    """
    init_k = tf.truncated_normal(kernel_params, mean=init_w[0], stddev=init_w[1])
    weights = tf.Variable(initial_value=init_k, name="weights")
    biases = tf.constant(0.0, tf.float32, [kernel_params[-1]], name="biases")
    conv = tf.nn.conv2d(input_tensor, weights, strides, padding=padding)
    value = tf.nn.bias_add(conv, biases)
    debug("C [%s]: %s c %s + %s = %s" % (scopename,str(input_tensor.shape), str(weights.shape), str(biases.shape), str(value.shape)))
    return tf.nn.relu(value, name = scopename)

def vec_seq_concat(seq_tensor, vec_tensor, sequence_length, order = 'vecfirst'):
    """
    concatenate each vector in vec_tensor to each element in seq_tensor, wrt the sequence length
    :param seq_tensor:
    :param vec_tensor:
    :param sequence_length:
    :param order:
    :return:
    """
    vec_dim = int(vec_tensor.shape[-1])
    # repeat the vec tensor to the sequence length
    vec_tensor = print_tensor(vec_tensor, "original vec tensor")
    seq_tensor = print_tensor(seq_tensor, "seq tensor")
    vec_tensor = tf.tile(vec_tensor, [1, sequence_length])
    vec_tensor = print_tensor(vec_tensor, "tiled tensor")

    # restore to one vector per column
    vec_tensor = tf.reshape(vec_tensor, [-1, vec_dim])
    vec_tensor = print_tensor(vec_tensor, "reshaped tiled tensor")
    # hor. concat with the seq_tensor
    if order == 'vecfirst':
        res = tf.concat([vec_tensor, seq_tensor],axis=1)
    else:
        res = tf.concat([seq_tensor, vec_tensor],axis=1)
    vec_tensor = print_tensor(vec_tensor, "concatted tensor")
    return res

def aggregate_clip_vectors(encoded_frames, encoded_dim, fpc, fusion_method):
    debug("Aggregating clip vectors, fpc:%d, dim:%d, inputshape:%s" % (fpc, encoded_dim, str(encoded_frames.shape)))
    encoded_frames = tf.reshape(encoded_frames, (-1, fpc, encoded_dim),
                                name="aggregate_clips")

    encoded_frames = print_tensor(encoded_frames, "Reshaped vectors")
    encoded_frames = apply_temporal_fusion(encoded_frames, encoded_dim, fpc, fusion_method)
    return encoded_frames


def apply_tensor_list_fusion(inputs, fusion_method, dims, fpcs, cpvs):
    if fusion_method == defs.fusion_method.avg:
        return tf.reduce_mean(inputs, axis=0), dims[[0], fpcs[0], cpvs[0]
    elif fusion_method == defs.fusion_method.concat:
        return tf.concat(inputs, axis=1), sum(dims), fpcs[0], cpvs[0]

    else if defs.check(fusion_method, defs.dual_fusion_method, do_boolean = True):
        if len(inputs) != 2:
            error("Requested dual fusion but supplied %d inputs" % len(inputs))
        mdim, adim = dims
        mfpc, afpc = fpcs
        mcpv, acpv = cpvs
        tile_num = int(mcpv/acpv)
        main, aux = inputs
        # duplicate aux to required fpc, if necessary
        aux = replicate_auxilliary_tensor(aux, adim, tile_aux )

        if fusion_method == defs.dual_fusion_method.ibias:
            # reshape seq vector to numclips x fpc x dim
            main= tf.reshape(input1, [-1, mfpc, mdim])
            main = print_tensor(main,"reshaped seq")
            # reshape the aux vectors to batch_size x fpc=1 x dim
            aux = tf.reshape(aux, [-1, 1, adim])
            aux = print_tensor(aux,"reshaped bias")
            # insert the aux as the first item in the seq - may need tf.expand on the fused
            combo = tf.concat([aux, main], axis=1)
            # increase the seq len to account for the input bias extra timestep
            combo_fpc = mfpc + 1
            info("Input bias augmented fpc: %d + 1 = %d" % (fpc1, combo_fpc))
            # restore to batchsize*seqlen x embedding_dim
            combo = tf.reshape(combo ,[-1, mdim])
            return combo, mdim, combo_fpc, mcpv

        elif fusion_method == defs.dual_fusion_method.concat:
            return  vec_seq_concat(main, aux, mfpc), mdim+adim, mfpc, mcpv
        else:
            error("Unknown dual fusion method: [%s]" % fusion_method)
    else:
        error("Unsupported tensor list aggregation method: [%s]" % fusion_method)

def replicate_auxilliary_tensor(input, dim, tile_num):
    # replicate each item in the input <tile_num> times, in place
    if tile_num > 1:
        input = tf.reshape(input, [1, -1])
        input = tf.tile(input, [tile_num, 1])
        input = tf.reshape(input, [-1, dim])
    return input

