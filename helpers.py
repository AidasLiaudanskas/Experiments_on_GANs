"""
All the functions that don't fit in anywhere else or reduce readability.
"""

import tensorflow as tf
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.celebA_64x64  # This is where the input_pipe is
# import tflib.small_imagenet
import tflib.ops.layernorm
import tflib.plot
import numpy as np
from glob import glob
import os
import functools
FLAGS = tf.app.flags.FLAGS
# print(FLAGS.gan_version)
# BATCH_SIZE = FLAGS.batch_size


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name + '.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name + '.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

# BCHW --> BHWC Mapping for Batchnorm is done in the function definition deeper in tflib.


def Batchnorm(name, axes, inputs):
    if ('Discriminator' in name) and (FLAGS.gan_version == 'wgan-gp'):
        if axes != [0, 2, 3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name, axes, inputs, fused=True)


def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)


def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4 * kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    return output


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1 = functools.partial(
            lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b = functools.partial(
            lib.ops.conv2d.Conv2D, input_dim=input_dim // 2, output_dim=output_dim // 2, stride=2)
        conv_2 = functools.partial(
            lib.ops.conv2d.Conv2D, input_dim=output_dim // 2, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = SubpixelConv2D
        conv_1 = functools.partial(
            lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b = functools.partial(
            lib.ops.deconv2d.Deconv2D, input_dim=input_dim // 2, output_dim=output_dim // 2)
        conv_2 = functools.partial(
            lib.ops.conv2d.Conv2D, input_dim=output_dim // 2, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1 = functools.partial(
            lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim // 2)
        conv_1b = functools.partial(
            lib.ops.conv2d.Conv2D, input_dim=input_dim // 2,  output_dim=output_dim // 2)
        conv_2 = functools.partial(
            lib.ops.conv2d.Conv2D, input_dim=input_dim // 2, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)

    output = conv_1(name + '.Conv1', filter_size=1,
                    inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_1b(name + '.Conv1B', filter_size=filter_size,
                     inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_2(name + '.Conv2', filter_size=1, inputs=output,
                    he_init=he_init, weightnorm=False, biases=False)
    output = Batchnorm(name + '.BN', [0, 2, 3], output)

    return shortcut + (0.3 * output)


def prepare_noise_samples(devices, Generator):
    fixed_noise = tf.constant(np.random.normal(
        size=(FLAGS.batch_size, 128)).astype('float32'))
    all_fixed_noise_samples = []
    for device_index, device in enumerate(devices):
        n_samples = FLAGS.batch_size // len(devices)
        all_fixed_noise_samples.append(Generator(
            n_samples, noise=fixed_noise[device_index * n_samples:(device_index + 1) * n_samples]))
    all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
    # print(all_fixed_noise_samples)
    return all_fixed_noise_samples


def get_dataset_files():
    pattern = os.path.join(FLAGS.dataset_path, FLAGS.dataset + ".tfrecords")
    files = glob(pattern)
    assert len(
        files) > 0, "Did not find any tfrecords files in the dataset_path folder"
    train_data_list = []
    for entity in files:
        train_data_list.append(entity)
    # Addition for multiple files:
    # images_per_file = FLAGS.images_per_file
    # if "celebA" in FLAGS.dataset:
    #     images_per_file = 206000
    # elif "profile_imgs_tfrecords" in FLAGS.dataset:
    #     images_per_file = 470000
    # elif "mnist" in FLAGS.dataset:
    #     images_per_file = 60000
    # # print (train_data_list)
    return train_data_list


def refresh_dirs(SUMMARY_DIR, OUTPUT_DIR, SAVE_DIR, restore):
    ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
    restore = restore and ckpt and ckpt.model_checkpoint_path
    if not restore:
        if tf.gfile.Exists(SUMMARY_DIR):
            tf.gfile.DeleteRecursively(SUMMARY_DIR)
            print("Log directory reconstructed")
        tf.gfile.MakeDirs(SUMMARY_DIR)

        if tf.gfile.Exists(SAVE_DIR):
            tf.gfile.DeleteRecursively(SAVE_DIR)
            print("Save directory reconstructed")
        tf.gfile.MakeDirs(SAVE_DIR)

        if tf.gfile.Exists(OUTPUT_DIR):
            tf.gfile.DeleteRecursively(OUTPUT_DIR)
            print("Output directory reconstructed")
        tf.gfile.MakeDirs(OUTPUT_DIR)


def generate_image(iteration, sess, output_dir, all_fixed_noise_samples, Generator, summary_writer):
    # add image to summary
    height = FLAGS.height
    if FLAGS.data_format == "NCHW":
        output_shape = [FLAGS.batch_size,
                        FLAGS.n_ch, height, height]
    else:
        output_shape = [FLAGS.batch_size,
                        height, height, FLAGS.n_ch]
    samples_reshaped = tf.reshape(
        all_fixed_noise_samples, output_shape)
    if FLAGS.data_format == "NCHW":
        # NCHW --> NHWC
        samples_reshaped = tf.transpose(samples_reshaped, [0, 2, 3, 1])
    image_op = tf.summary.image(
        'generator output', samples_reshaped)
    image_summary, samples = sess.run(
        [image_op, all_fixed_noise_samples])
    # summary_writer.add_summary(image_summary, iteration)
    # samples = sess.run(all_fixed_noise_samples)

    samples = ((samples + 1.) * (255.99 / 2)).astype('int32')
    samples = np.reshape(samples, output_shape)
    if FLAGS.data_format == "NHWC" and samples.shape[3] in [1, 3]:
        # NHWC --> NCHW
        samples = np.transpose(samples, [0, 3, 1, 2])
    lib.save_images.save_images(
        samples, output_dir + '/samples_{}.png'.format(iteration))


"""
Making a sample of how the training data looks like
"""


def sample_dataset(sess, all_real_data_conv, output_dir):
    _x_r = sess.run(all_real_data_conv)
    # print(_x_r[0])
    if np.amin(_x_r) < 0:
        _x_r = ((_x_r + 1.) * (255.99 / 2)).astype('int32')
        # print("Took path less than zero")
    else:
        _x_r = (_x_r).astype('int32')
    # print("Dataset was sampled")
    if FLAGS.data_format == "NHWC":
        lib.save_images.save_images(np.transpose(_x_r.reshape(
            (FLAGS.batch_size, FLAGS.height, FLAGS.height, FLAGS.n_ch)), (0,3,1,2)), output_dir + '/samples_groundtruth.png')
    else:
        lib.save_images.save_images(_x_r.reshape(
            (FLAGS.batch_size, FLAGS.n_ch, FLAGS.height, FLAGS.height)), output_dir + '/samples_groundtruth.png')
