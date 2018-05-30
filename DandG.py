import tensorflow as tf
import tflib as lib
import functools
FLAGS = tf.app.flags.FLAGS
from helpers import *
from tensorflow.contrib import slim

# OUTPUT_DIM = FLAGS.FLAGS.output_dim
# MODE = FLAGS.gan_version
# FLAGS.model_dim = FLAGS.model_dim


def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """

    # For actually generating decent samples, use this one
    # return GoodGenerator, GoodDiscriminator

    # Baseline (G: DCGAN, D: DCGAN)
    if FLAGS.architecture.lower() == "dcgan":
        return DCGANGenerator, DCGANDiscriminator

    # No BN and constant number of filts in G
    # return WGANPaper_CrippledDCGANGenerator, DCGANDiscriminator

    # 512-dim 4-layer ReLU MLP G
    # return FCGenerator, DCGANDiscriminator

    # No normalization anywhere
    # return functools.partial(DCGANGenerator, bn=False), functools.partial(DCGANDiscriminator, bn=False)

    # Gated multiplicative nonlinearities everywhere
    # return MultiplicativeDCGANGenerator, MultiplicativeDCGANDiscriminator

    # tanh nonlinearities everywhere
    # return functools.partial(DCGANGenerator, bn=True, nonlinearity=tf.tanh), \
    #        functools.partial(DCGANDiscriminator, bn=True, nonlinearity=tf.tanh)

    # 101-layer ResNet G and D
    if FLAGS.architecture.lower() == "resnet":
        return Resnet101Generator, Resnet101Discriminator

    raise Exception('You must choose an architecture!')


# ! Generators

def GoodGenerator(n_samples, noise=None, dim=FLAGS.model_dim, nonlinearity=tf.nn.relu):

    n_samples = int(n_samples)
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear(
        'Generator.Input', 128, 4 * 4 * 8 * dim, noise)
    output = tf.reshape(output, [-1, 8 * dim, 4, 4])

    output = ResidualBlock('Generator.Res1', 8 * dim,
                           8 * dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res2', 8 * dim,
                           4 * dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res3', 4 * dim,
                           2 * dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res4', 2 * dim,
                           1 * dim, 3, output, resample='up')

    output = Batchnorm('Generator.OutputN', [0, 2, 3], output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1 * dim, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, FLAGS.output_dim])


def FCGenerator(n_samples, noise=None, FC_DIM=512):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = ReLULayer('Generator.1', 128, FC_DIM, noise)
    output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Generator.Out', FC_DIM, FLAGS.output_dim, output)

    output = tf.tanh(output)

    return output


def MultiplicativeDCGANGenerator(n_samples, noise=None, dim=FLAGS.model_dim, bn=True):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear(
        'Generator.Input', 128, 4 * 4 * 8 * dim * 2, noise)
    output = tf.reshape(output, [-1, 8 * dim * 2, 4, 4])
    if bn:
        output = Batchnorm('Generator.BN1', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = lib.ops.deconv2d.Deconv2D(
        'Generator.2', 8 * dim, 4 * dim * 2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = lib.ops.deconv2d.Deconv2D(
        'Generator.3', 4 * dim, 2 * dim * 2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = lib.ops.deconv2d.Deconv2D(
        'Generator.4', 2 * dim, dim * 2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, FLAGS.output_dim])


def WGANPaper_CrippledDCGANGenerator(n_samples, noise=None, dim=FLAGS.model_dim):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4 * 4 * dim, noise)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, dim, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, FLAGS.output_dim])


def DCGANGenerator(n_samples, noise=None, dim=FLAGS.model_dim, bn=True, nonlinearity=tf.nn.relu):
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear(
        'Generator.Input', 128, 4 * 4 * 8 * dim, noise)
    output = tf.reshape(output, [-1, 8 * dim, 4, 4])
    if bn:
        output = Batchnorm('Generator.BN1', [0, 2, 3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D(
        'Generator.2', 8 * dim, 4 * dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0, 2, 3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D(
        'Generator.3', 4 * dim, 2 * dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0, 2, 3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2 * dim, dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0, 2, 3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, FLAGS.output_dim])

# Too powerful of a discriminator
# Send TFRecords input to other people
# Gan usecases - speech, facerec
# Gradient penalty for generator only maybe? Overpowered Generator


def Resnet101Generator(n_samples, noise=None, dim=FLAGS.model_dim):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear(
        'Generator.Input', 128, 4 * 4 * 8 * dim, noise)
    output = tf.reshape(output, [-1, 8 * dim, 4, 4])

    for i in range(6):
        output = ResidualBlock('Generator.4x4_{}'.format(
            i), 8 * dim, 8 * dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up1', 8 * dim,
                           4 * dim, 3, output, resample='up')
    for i in range(6):
        output = ResidualBlock('Generator.8x8_{}'.format(
            i), 4 * dim, 4 * dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up2', 4 * dim,
                           2 * dim, 3, output, resample='up')
    for i in range(6):
        output = ResidualBlock('Generator.16x16_{}'.format(
            i), 2 * dim, 2 * dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up3', 2 * dim,
                           1 * dim, 3, output, resample='up')
    for i in range(6):
        output = ResidualBlock('Generator.32x32_{}'.format(
            i), 1 * dim, 1 * dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up4', 1 * dim,
                           dim // 2, 3, output, resample='up')
    for i in range(5):
        output = ResidualBlock('Generator.64x64_{}'.format(
            i), dim // 2, dim // 2, 3, output, resample=None)

    output = lib.ops.conv2d.Conv2D(
        'Generator.Out', dim // 2, 3, 1, output, he_init=False)
    output = tf.tanh(output / 5.)

    return tf.reshape(output, [-1, FLAGS.output_dim])


# ! Discriminators


def MultiplicativeDCGANDiscriminator(inputs, dim=FLAGS.model_dim, bn=True):
    output = tf.reshape(inputs, [-1, 3, dim, dim])

    output = lib.ops.conv2d.Conv2D(
        'Discriminator.1', 3, dim * 2, 5, output, stride=2)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = lib.ops.conv2d.Conv2D(
        'Discriminator.2', dim, 2 * dim * 2, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN2', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = lib.ops.conv2d.Conv2D(
        'Discriminator.3', 2 * dim, 4 * dim * 2, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = lib.ops.conv2d.Conv2D(
        'Discriminator.4', 4 * dim, 8 * dim * 2, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    output = lib.ops.linear.Linear(
        'Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

    return tf.reshape(output, [-1])


def FCDiscriminator(inputs, FC_DIM=512, n_layers=3):
    output = LeakyReLULayer('Discriminator.Input', FLAGS.output_dim, FC_DIM, inputs)
    for i in range(n_layers):
        output = LeakyReLULayer(
            'Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Discriminator.Out', FC_DIM, 1, output)

    return tf.reshape(output, [-1])


def GoodDiscriminator(inputs, dim=FLAGS.model_dim):
    output = tf.reshape(inputs, [-1, 3, dim, dim])
    output = lib.ops.conv2d.Conv2D(
        'Discriminator.Input', 3, dim, 3, output, he_init=False)

    output = ResidualBlock('Discriminator.Res1', dim,
                           2 * dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res2', 2 * dim,
                           4 * dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res3', 4 * dim,
                           8 * dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res4', 8 * dim,
                           8 * dim, 3, output, resample='down')

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    output = lib.ops.linear.Linear(
        'Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

    return tf.reshape(output, [-1])


def Resnet101Discriminator(inputs, dim=FLAGS.model_dim):
    output = tf.reshape(inputs, [-1, 3, dim, dim])
    output = lib.ops.conv2d.Conv2D(
        'Discriminator.In', 3, dim // 2, 1, output, he_init=False)

    for i in range(5):
        output = ResidualBlock('Discriminator.64x64_{}'.format(
            i), dim // 2, dim // 2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down1', dim // 2,
                           dim * 1, 3, output, resample='down')
    for i in range(6):
        output = ResidualBlock('Discriminator.32x32_{}'.format(
            i), dim * 1, dim * 1, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down2', dim * 1,
                           dim * 2, 3, output, resample='down')
    for i in range(6):
        output = ResidualBlock('Discriminator.16x16_{}'.format(
            i), dim * 2, dim * 2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down3', dim * 2,
                           dim * 4, 3, output, resample='down')
    for i in range(6):
        output = ResidualBlock('Discriminator.8x8_{}'.format(
            i), dim * 4, dim * 4, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down4', dim * 4,
                           dim * 8, 3, output, resample='down')
    for i in range(6):
        output = ResidualBlock('Discriminator.4x4_{}'.format(
            i), dim * 8, dim * 8, 3, output, resample=None)

    pre_output = output = tf.reshape(output, [-1, 4 * 4 * dim])
    output = lib.ops.linear.Linear(
        'Discriminator.Output', 4 * 4 * dim, 1, output)

    # TODO originally was 4*4*8*dim
    # Try decreasing by 8
    return tf.reshape(output / 5., [-1]), pre_output


def DCGANDiscriminator(inputs, dim=FLAGS.model_dim, bn=True, nonlinearity=LeakyReLU):
    if FLAGS.data_format == "NHWC":
        output = tf.reshape(inputs, [-1, dim, dim, 3])
    else:
        output = tf.reshape(inputs, [-1, 3, dim, dim])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    output = lib.ops.conv2d.Conv2D(
        'Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D(
        'Discriminator.2', dim, 2 * dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN2', [0, 2, 3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D(
        'Discriminator.3', 2 * dim, 4 * dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0, 2, 3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D(
        'Discriminator.4', 4 * dim, 8 * dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0, 2, 3], output)
    output = nonlinearity(output)

    pre_output = output = tf.reshape(output, [-1, 4 * 4  * dim])
    # Originally was 4*4*8*dim
    # Now decreased to 4*4*dim = 1024 for dim=64.
    # Should be more meaningful as the image has only 4k pixels
    output = lib.ops.linear.Linear(
        'Discriminator.Output', 4 * 4 * dim, 1, output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1]), pre_output


# BEGAN Additions:
# Taken from https://github.com/carpedm20/BEGAN-tensorflow/blob/master/models.py

def GeneratorCNN(z, reuse, output_num, repeat_num, data_format=FLAGS.data_format, hidden_num=64):
    with tf.variable_scope("Generator", reuse=reuse) as vs:
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)

        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1,
                            activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1,
                            activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, 3, 3, 1, activation_fn=None,
                          data_format=data_format)

    # variables = tf.contrib.framework.get_variables(vs)
    return out


def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num=64, data_format=FLAGS.data_format):
    with tf.variable_scope("Discriminator") as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1,
                        activation_fn=tf.nn.elu, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1,
                            activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1,
                            activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2,
                                activation_fn=tf.nn.elu, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

        # Decoder
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(x, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)

        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1,
                            activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1,
                            activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, input_channel, 3, 1,
                          activation_fn=None, data_format=data_format)

    # variables = tf.contrib.framework.get_variables(vs)
    # Can return above if needed
    return out, z


# Report of what I've done so far;
# Try cloud cluster
#
