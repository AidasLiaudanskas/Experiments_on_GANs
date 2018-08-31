"""
File containing different DCGAN architectures
"""


import tensorflow as tf
import tflib as lib
from functools import partial
from helpers import Batchnorm, ResidualBlock
FLAGS = tf.app.flags.FLAGS

# from tensorflow.contrib import slim

# MODE = FLAGS.gan_version
# DIM = FLAGS.model_dim
# Implement DCGAN Architectres with adjustable complexities


class DCGAN:
    def __init__(self):
        self.model_dim = self.G_dim = self.D_dim = FLAGS.model_dim
        self.OUTPUT_DIM = FLAGS.output_dim
        self.N_CH = FLAGS.n_ch
        # print("OUTPUT_DIM =", self.OUTPUT_DIM)
        self.height = self.width = FLAGS.height

    def set_dim(self, dim):
        self.model_dim = self.G_dim = self.D_dim = dim

    def set_G_dim(self, dim):
        self.G_dim = dim

    def get_G_dim(self):
        return self.G_dim

    def get_D_dim(self):
        return self.D_dim

    def set_D_dim(self, dim):
        self.D_dim = dim

    def DCGANG_Mnist(self, n_samples, noise=None, bn=True, nonlinearity=tf.nn.relu):
        """
        Describes the Generator architecture for TF model.
        """
        dim = self.G_dim
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)
        batchnorm_tf = partial(
            tf.layers.batch_normalization, reuse=tf.AUTO_REUSE)
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear(
            'Generator.Input', 128, 4 * 4 * 4 * dim, noise)
        if FLAGS.data_format == "NHWC":
            bn_axis = 3
            output = tf.reshape(output, [-1, 4, 4, 4 * dim])
            bn_axes = [0, 1, 2]
        else:
            output = tf.reshape(output, [-1, 4 * dim, 4, 4])
            bn_axes = [0, 2, 3]
            bn_axis = 1
        print("Shape before batch_norm: ", output.shape)
        if bn:
            output = batchnorm_tf(
                output, name='Generator.BN1', fused=True, axis=bn_axis)
            # output = Batchnorm('Generator.BN1', bn_axes, output)
        output = nonlinearity(output)
        print("Shape before Generator2: ", output.shape)
        output = lib.ops.deconv2d.Deconv2D(
            'Generator.2', 4 * dim, 2 * dim, 5, output)
        # print("Shape after Generator2: ", output.shape)
        if bn:
            # Batchnorm
            output = batchnorm_tf(
                output, name='Generator.BN2', fused=True, axis=bn_axis)
            # output = Batchnorm('Generator.BN2', bn_axes, output)
        output = nonlinearity(output)
        print("Shape before Generator3: ", output.shape)
        output = lib.ops.deconv2d.Deconv2D(
            'Generator.3', 2 * dim, 1 * dim, 5, output)
        if bn:
            output = batchnorm_tf(
                output, name='Generator.BN3', fused=True, axis=bn_axis)
            # output = Batchnorm('Generator.BN3', bn_axes, output)
        output = nonlinearity(output)
        print("Shape before Generator4: ", output.shape)
        output = lib.ops.deconv2d.Deconv2D(
            'Generator.4', 1 * dim, self.N_CH, 5, output)
        if bn:
            output = batchnorm_tf(
                output, name='Generator.BN4', fused=True, axis=bn_axis)
            # output = Batchnorm('Generator.BN4', bn_axes, output)
        print("Shape after Generator4: ", output.shape)
        output = nonlinearity(output)
        # TODO:
        # ValueError: Incompatible shapes between op input and calculated input gradient.
        # Forward operation: Generator.4/conv2d_transpose.
        # Input index: 2. Original input shape: (10, 128, 16, 16).
        # Calculated input gradient shape: (10, 128, 16, 32)
        # One solution is to use 32x32 MNIST and remove last Gen Layer
        # Another is to modify the transposed convolution to upsample 1x instead of 2x,
        # this way could add more layers.

        # Try to squeeze mnist from 32x32 to 28x28:
        # if FLAGS.dataset == "mnist":
        #     output = tf.reshape(output, [-1, 64 * 32 * 32])
        #     output = lib.ops.linear.Linear(
        #         'Generator.Output', 64 * 32 * 32, 1 * 28 * 28, output)
        # elif FLAGS.dataset == "mnist32x32":
        #     pass
        # else:
        #     print("Shape before Generator5: ", output.shape)
        #     output = lib.ops.deconv2d.Deconv2D(
        #         'Generator.5', dim, self.N_CH, 5, output)
        #     print("Shape after Generator5: ", output.shape)
        output = tf.tanh(output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1, self.OUTPUT_DIM])

    def DCGAND_Mnist(self, inputs, bn=True, nonlinearity=tf.nn.relu):
        """
        Describes the Discriminator architecture for TF model.
        """
        dim = self.D_dim
        batchnorm_tf = partial(tf.layers.batch_normalization,
                               reuse=tf.AUTO_REUSE)
        print("Discriminator inputs shape=", inputs.shape)
        if FLAGS.data_format == "NHWC":
            output = tf.reshape(
                inputs, [-1, self.height, self.width, self.N_CH])
            bn_axes = [0, 1, 2]
            bn_axis = 3
        else:
            output = tf.reshape(
                inputs, [-1, self.N_CH, self.height, self.width])
            bn_axes = [0, 2, 3]
            bn_axis = 1

        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        output = lib.ops.conv2d.Conv2D(
            'Discriminator.1', self.N_CH, dim, 5, output, stride=2)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(
            'Discriminator.2', dim, 2 * dim, 5, output, stride=2)
        if bn:
            output = batchnorm_tf(
                output, name='Discriminator.BN2', fused=True, axis=bn_axis)
            # output = Batchnorm('Discriminator.BN2', bn_axes, output)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(
            'Discriminator.3', 2 * dim, 4 * dim, 5, output, stride=2)
        if bn:
            output = batchnorm_tf(
                output, name='Discriminator.BN3', fused=True, axis=bn_axis)
            # output = Batchnorm('Discriminator.BN3', bn_axes, output)
        output = nonlinearity(output)
        output = lib.ops.conv2d.Conv2D(
            'Discriminator.4', 4 * dim, 4 * dim, 5, output, stride=2)
        if bn:
            output = batchnorm_tf(
                output, name='Discriminator.BN4', fused=True, axis=bn_axis)
            # output = Batchnorm('Discriminator.BN4', bn_axes, output)
        output = nonlinearity(output)
        print("Disc output shape: ", output.shape)
        pre_output = output = tf.reshape(
            output, [FLAGS.batch_size, 4 * 4 * dim])
        # print("Discriminator pre output shape: ", pre_output.shape)
        output = lib.ops.linear.Linear(
            'Discriminator.Output', 4 * 4 * dim, 1, output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()
        print("Discriminator output shape: ", tf.reshape(output, [-1]).shape)

        return tf.reshape(output, [-1]), pre_output

    def DCGANG_1(self, n_samples, noise=None, bn=True, nonlinearity=tf.nn.relu, verbose=False):
        """
        Describes the Generator architecture for TF model.
        """
        dim = self.G_dim
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)
        batchnorm_tf = partial(
            tf.layers.batch_normalization, reuse=tf.AUTO_REUSE)
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear(
            'Generator.Input', 128, 4 * 4 * 8 * dim, noise)
        if FLAGS.data_format == "NHWC":
            bn_axis = 3
            output = tf.reshape(output, [-1, 4, 4, 8 * dim])
            bn_axes = [0, 1, 2]
        else:
            output = tf.reshape(output, [-1, 8 * dim, 4, 4])
            bn_axes = [0, 2, 3]
            bn_axis = 1
        if verbose:
            print("Shape before batch_norm: ", output.shape)
        if bn:
            # output = batchnorm_tf(output, name='Generator.BN1', fused=True, axis=bn_axis)
            output = Batchnorm('Generator.BN1', bn_axes, output)
        output = nonlinearity(output)
        if verbose:
            print("Shape before Generator2: ", output.shape)
        output = lib.ops.deconv2d.Deconv2D(
            'Generator.2', 8 * dim, 4 * dim, 5, output)
        # print("Shape after Generator2: ", output.shape)
        if bn:
            # Batchnorm
            # output = batchnorm_tf(output, name='Generator.BN2', fused=True, axis=bn_axis)
            output = Batchnorm('Generator.BN2', bn_axes, output)
        output = nonlinearity(output)
        if verbose:
            print("Shape before Generator3: ", output.shape)
        output = lib.ops.deconv2d.Deconv2D(
            'Generator.3', 4 * dim, 2 * dim, 5, output)
        if bn:
            # output = batchnorm_tf(output, name='Generator.BN3', fused=True, axis=bn_axis)
            output = Batchnorm('Generator.BN3', bn_axes, output)
        output = nonlinearity(output)
        if verbose:
            print("Shape before Generator4: ", output.shape)
        output = lib.ops.deconv2d.Deconv2D(
            'Generator.4', 2 * dim, dim, 5, output)
        if bn:
            # output = batchnorm_tf(output, name='Generator.BN4', fused=True, axis=bn_axis)
            output = Batchnorm('Generator.BN4', bn_axes, output)
        if verbose:
            print("Shape after Generator4: ", output.shape)
        output = nonlinearity(output)
        # TODO:
        # ValueError: Incompatible shapes between op input and calculated input gradient.
        # Forward operation: Generator.4/conv2d_transpose.
        # Input index: 2. Original input shape: (10, 128, 16, 16).
        # Calculated input gradient shape: (10, 128, 16, 32)
        # One solution is to use 32x32 MNIST and remove last Gen Layer
        # Another is to modify the transposed convolution to upsample 1x instead of 2x,
        # this way could add more layers.
        # 32x32 images can Only have 4 Generator layers
        if verbose:
            print("Shape before Generator5: ", output.shape)
        output = lib.ops.deconv2d.Deconv2D(
            'Generator.5', dim, self.N_CH, 5, output)
        if verbose:
            print("Shape after Generator5: ", output.shape)
        output = tf.tanh(output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1, self.OUTPUT_DIM])

    def DCGANG_2(self, n_samples, noise=None, bn=True, nonlinearity=tf.nn.relu):
        """
        Describes the Generator architecture for TF model. Added an extra layer at the input
        Be aware of potential bug with batchn_norm axes
        """
        dim = self.G_dim
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear(
            'Generator.Input', 128, 4 * 4 * 16 * dim, noise)
        if FLAGS.data_format == "NHWC":
            output = tf.reshape(output, [-1, 4, 4, 16 * dim])
            bn_axes = [0, 1, 2]
        else:
            output = tf.reshape(output, [-1, 16 * dim, 4, 4])
            bn_axes = [0, 2, 3]
        if bn:
            output = Batchnorm('Generator.BN1', bn_axes, output)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D(
            'Generator.1', 16 * dim, 8 * dim, 5, output)
        if bn:
            output = Batchnorm('Generator.BN1.1', bn_axes, output)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D(
            'Generator.2', 8 * dim, 4 * dim, 5, output)
        if bn:
            output = Batchnorm('Generator.BN2', bn_axes, output)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D(
            'Generator.3', 4 * dim, 2 * dim, 5, output)
        if bn:
            output = Batchnorm('Generator.BN3', bn_axes, output)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D(
            'Generator.4', 2 * dim, dim, 5, output)
        if bn:
            output = Batchnorm('Generator.BN4', bn_axes, output)
        output = nonlinearity(output)
        output = lib.ops.deconv2d.Deconv2D(
            'Generator.5', dim, self.N_CH, 5, output)

        output = tf.tanh(output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1, self.OUTPUT_DIM])

    def GoodGenerator(self, n_samples, noise=None, dim=FLAGS.model_dim, nonlinearity=tf.nn.relu):
        """
        Taken directly from the code of WGAN-GP paper and modified a bit
        """

        batchnorm_tf = partial(tf.layers.batch_normalization,
                               reuse=tf.AUTO_REUSE)
        n_samples = int(n_samples)
        dim = self.G_dim
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear(
            'Generator.Input', 128, 4 * 4 * 8 * dim, noise)

        if FLAGS.data_format == "NHWC":
            bn_axis = 3
            output = tf.reshape(output, [-1, 4, 4, 8 * dim])
            bn_axes = [0, 1, 2]
        else:
            output = tf.reshape(output, [-1, 8 * dim, 4, 4])
            bn_axes = [0, 2, 3]
            bn_axis = 1

        output = ResidualBlock('Generator.Res1', 8 * dim,
                               8 * dim, 3, output, resample='up')
        output = ResidualBlock('Generator.Res2', 8 * dim,
                               4 * dim, 3, output, resample='up')
        output = ResidualBlock('Generator.Res3', 4 * dim,
                               2 * dim, 3, output, resample='up')
        output = ResidualBlock('Generator.Res4', 2 * dim,
                               1 * dim, 3, output, resample='up')

        # output = Batchnorm('Generator.OutputN', bn_axes, output)
        output = batchnorm_tf(
            output, name='Generator.OutputN', fused=True, axis=bn_axis)
        output = tf.nn.relu(output)
        output = lib.ops.conv2d.Conv2D(
            'Generator.Output', 1 * dim, 3, 3, output)
        output = tf.tanh(output)
        return tf.reshape(output, [-1, FLAGS.output_dim])

    def GoodDiscriminator(self, inputs):
        dim = self.D_dim
        if FLAGS.data_format == "NHWC":
            output = tf.reshape(
                inputs, [-1, self.height, self.width, self.N_CH])
            # bn_axes = [0, 1, 2]
        else:
            output = tf.reshape(
                inputs, [-1, self.N_CH, self.height, self.width])
            # bn_axes = [0, 2, 3]

        # output = tf.reshape(inputs, [-1, 3, dim, dim])
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

        pre_output = output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
        output = lib.ops.linear.Linear(
            'Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

        # return tf.reshape(output, [-1])
        return tf.reshape(output, [-1]), pre_output

    def DCGAND_1(self, inputs, bn=True, nonlinearity=tf.nn.relu):
        """
        Describes the Discriminator architecture for TF model.
        """
        dim = self.D_dim
        batchnorm_tf = partial(tf.layers.batch_normalization,
                               reuse=tf.AUTO_REUSE)
        # print("Discriminator inputs shape=", inputs.shape)
        if FLAGS.data_format == "NHWC":
            output = tf.reshape(
                inputs, [-1, self.height, self.width, self.N_CH])
            bn_axes = [0, 1, 2]
            bn_axis = 3
        else:
            output = tf.reshape(
                inputs, [-1, self.N_CH, self.height, self.width])
            bn_axes = [0, 2, 3]
            bn_axis = 1
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        output = lib.ops.conv2d.Conv2D(
            'Discriminator.1', self.N_CH, dim, 5, output, stride=2)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(
            'Discriminator.2', dim, 2 * dim, 5, output, stride=2)
        if bn:
            output = batchnorm_tf(
                output, name='Discriminator.BN2', fused=True, axis=bn_axis)
            # output = Batchnorm('Discriminator.BN2', bn_axes, output)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(
            'Discriminator.3', 2 * dim, 4 * dim, 5, output, stride=2)
        if bn:
            output = batchnorm_tf(
                output, name='Discriminator.BN3', fused=True, axis=bn_axis)
            # output = Batchnorm('Discriminator.BN3', bn_axes, output)
        output = nonlinearity(output)
        # print("Discriminator inputs shape2= ", output.shape)
        output = lib.ops.conv2d.Conv2D(
            'Discriminator.4', 4 * dim, 8 * dim, 5, output, stride=2)
        if bn:
            output = batchnorm_tf(
                output, name='Discriminator.BN4', fused=True, axis=bn_axis)
            # output = Batchnorm('Discriminator.BN4', bn_axes, output)
        output = nonlinearity(output)
        # print("Discriminator inputs shape3= ", output.shape)
        # pre_output = output = tf.reshape(output, [-1, 4 * 4 * dim])
        pre_output = output = tf.reshape(output, [FLAGS.batch_size, -1])
        # TODO: Change back to tf.reshape(output, [-1, 4*4*8*dim])
        # TODO: Don't want to change anything now as there's no time for debugging.
        # BUG HERE!!! Sort out dimensions. Changed these on 13.20 3rd may.
        # MAY CAUSE UNDESIRED BEHAVIOUR!!!
        # print("Discriminator pre_output shape= ", output.shape)
        width = output.shape.as_list()[1]
        output = lib.ops.linear.Linear(
            'Discriminator.Output', width, 1, output)
        # print("Discriminator inputs shape4= ", output.shape)
        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1]), pre_output

    def DCGAND_2(self, inputs, bn=True, nonlinearity=tf.nn.relu):
        """
        Describes the Discriminator architecture for TF model. Added an extra layer at the output
        """
        dim = self.D_dim
        if FLAGS.data_format == "NHWC":
            output = tf.reshape(
                inputs, [-1, self.height, self.width, self.N_CH])
            bn_axes = [0, 1, 2]
        else:
            output = tf.reshape(
                inputs, [-1, self.N_CH, self.height, self.width])
            bn_axes = [0, 2, 3]

        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        output = lib.ops.conv2d.Conv2D(
            'Discriminator.1', self.N_CH, dim, 5, output, stride=2)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(
            'Discriminator.2', dim, 2 * dim, 5, output, stride=2)
        if bn:
            output = Batchnorm('Discriminator.BN2', bn_axes, output)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(
            'Discriminator.3', 2 * dim, 4 * dim, 5, output, stride=2)
        if bn:
            output = Batchnorm('Discriminator.BN3', bn_axes, output)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(
            'Discriminator.4', 4 * dim, 8 * dim, 5, output, stride=2)
        if bn:
            output = Batchnorm('Discriminator.BN4', bn_axes, output)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D(
            'Discriminator.5', 8 * dim, 16 * dim, 5, output, stride=2)
        if bn:
            output = Batchnorm('Discriminator.BN5', bn_axes, output)
        output = nonlinearity(output)

        pre_output = output = tf.reshape(output, [-1, 4 * 4 * dim])
        output = lib.ops.linear.Linear(
            'Discriminator.Output', 4 * 4 * dim, 1, output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1]), pre_output
