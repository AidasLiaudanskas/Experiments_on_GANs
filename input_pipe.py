"""
Abstraction functions to take care of the tfrecords format.

"""

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


def read_tfrecords(filename_queue, height, width, no_channels=3):
    reader = tf.TFRecordReader()
    key, record_string = reader.read(filename_queue)
    features = tf.parse_single_example(record_string, features={
        'image_raw': tf.FixedLenFeature([], tf.string)
    })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape(height * width * no_channels)
    image = tf.reshape(image, [height, width, no_channels])
    return image


def input_pipeline(filename_list, batch_size=32, num_threads=4, num_epochs=None, verbose=False):
    """
    Takes in tfrecord files with images coded as uint8 in range 0-255
    Outputs tensors coded as float32 in range -1 to 1 - not true!!
    changed to 0.0 - 255.0 output range as the GANs didn't converge otherwise.
    The input is cast to [-1;1] in split_and_setup_costs function
    """
    with tf.name_scope('pipeline'):
        num_threads = FLAGS.num_threads
        no_channels = FLAGS.n_ch
        height = width = FLAGS.height
        tf.local_variables_initializer()
        filename_queue = tf.train.string_input_producer(filename_list)
        # Even when reading in multiple threads, share the filename queue.
        image = read_tfrecords(filename_queue, height, width, no_channels)
        if verbose:
            print("Tfrecords image shape: ", image.shape)
        images = tf.train.shuffle_batch(
            [image], batch_size=batch_size, num_threads=num_threads,
            capacity=1000 + FLAGS.capacity_factor * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)
        # Return BCHW tensors:
        images = tf.cast(images, dtype=tf.float32)
        if verbose:
            print("images shape:", images.shape)
        if FLAGS.data_format == "NCHW":  # This format (NCHW) is faster on GPU
            images = tf.transpose(images, perm=[0, 3, 1, 2])
    return images
