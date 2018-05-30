import tflib as lib
import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
_default_weightnorm = False


def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True


_weights_stdev = None


def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev


def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None


def Deconv2D(
        name,
        input_dim,
        output_dim,
        filter_size,
        inputs,
        he_init=True,
        weightnorm=None,
        biases=True,
        gain=1.,
        mask_type=None,
):
    """
    inputs: tensor of shape (batch size, height, width, input_dim)
    returns: tensor of shape (batch size, 2*height, 2*width, output_dim)
    """
    with tf.name_scope(name) as scope:

        if mask_type != None:
            raise Exception('Unsupported configuration')

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        stride = 2
        fan_in = input_dim * filter_size**2 / (stride**2)
        fan_out = output_dim * filter_size**2

        if he_init:
            filters_stdev = np.sqrt(4. / (fan_in + fan_out))
        else:  # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2. / (fan_in + fan_out))

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )

        filter_values *= gain

        filters = lib.param(
            name + '.Filters',
            filter_values
        )

        if weightnorm == None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(
                np.sum(np.square(filter_values), axis=(0, 1, 3)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(
                    tf.square(filters), reduction_indices=[0, 1, 3]))
                filters = filters * tf.expand_dims(target_norms / norms, 1)

        input_shape = inputs.get_shape().as_list()
        # input_shapes = [inputs.shape[i] for i in range(4)]
        if FLAGS.data_format == "NHWC":
            # if inputs.shape[1] in [1,3]:
            #     inputs = tf.transpose(inputs, [0,2,3,1], name='NCHW_to_NHWC')
            output_shape = tf.stack(
                [input_shape[0], 2 * input_shape[1], 2 * input_shape[2], output_dim])
            # print("Took path NHWC")
        else:
            output_shape = tf.stack(
                [input_shape[0], output_dim, 2 * input_shape[2], 2 * input_shape[3]])
            # print("Took path NCHW")

        # try: # tf pre-1.0 (top) vs 1.0 (bottom)
        #     output_shape = tf.pack([input_shape[0], 2*input_shape[1], 2*input_shape[2], output_dim])
        # except Exception as e:
            # output_shape = tf.stack([input_shape[0], 2*input_shape[1], 2*input_shape[2], output_dim])

        # print("Deconv Input, filter, output shape: ", inputs.shape, filters.shape, output_shape.shape)
        # print("Pre conv2d_transpose: ", inputs.shape, [
              # input_shape[0], output_dim, 2 * input_shape[2], 2 * input_shape[3]])
        result = tf.nn.conv2d_transpose(
            value=inputs,
            filter=filters,
            output_shape=output_shape,
            strides=[1, 2, 2, 1],
            padding='SAME',
            data_format=FLAGS.data_format
        )
        # print("After conv2d_transpose: ", result.shape)
        if biases:
            _biases = lib.param(
                name + '.Biases',
                np.zeros(output_dim, dtype='float32')
            )
            result = tf.nn.bias_add(
                result, _biases, data_format=FLAGS.data_format)
        # if FLAGS.data_format=="NCHW" and result[3] in [1,3]:
        #     result = tf.transpose(result, [0,3,1,2], name='NHWC_to_NCHW')

        return result
