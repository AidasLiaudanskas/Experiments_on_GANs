#!/usr/bin/env python3

# import os
# import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import tensorflow as tf
import fid
# from scipy.misc import imread
import flags
import helpers
FLAGS = tf.app.flags.FLAGS
from input_pipe import input_pipeline
train_data_list = helpers.get_dataset_files()
batch_size = BATCH_SIZE = FLAGS.batch_size

# path for where to store the statistics
output_path = './tmp/' + FLAGS.dataset + '_stats.npz'
# output_path_kid = './tmp/' + FLAGS.dataset + '_activations.npz'
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = './tmp'
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(
    inception_path)  # download inception if necessary
print("ok")

print("create inception graph..", end=" ", flush=True)
# load the graph into the current TF graph
fid.create_inception_graph(inception_path)
print("ok")

# images = tf.squeeze(
#     tf.round((input_pipeline(train_data_list, batch_size=BATCH_SIZE) + 1) * 255))
images = tf.squeeze(((input_pipeline(train_data_list, batch_size=BATCH_SIZE))))
# Split data over multiple GPUs:
print("calculte FID stats..", end=" ", flush=True)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    try:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        no_images = (50000 // batch_size) * batch_size
        temp_image_list = np.empty([no_images, FLAGS.height, FLAGS.height, 3])
        # Build a massive array of dataset images
        for i in range(no_images // batch_size):
            np_images = np.squeeze(np.asarray(sess.run([images])))
            # print(np_images)
            # print("np_images shape", np_images.shape)
            temp_image_list[i * batch_size:(i + 1) *
                            batch_size, :, :, 0] = np_images
            temp_image_list[i * batch_size:(i + 1) *
                            batch_size, :, :, 1] = np_images
            temp_image_list[i * batch_size:(i + 1) *
                            batch_size, :, :, 2] = np_images
#            print(i)
            # break
        mu, sigma, activations_pool3, _ = fid.calculate_activation_statistics(
            temp_image_list, sess, batch_size=32, verbose=False)
        np.savez_compressed(output_path, mu=mu, sigma=sigma,
                            activations_pool3=activations_pool3)
    except Exception as e:
        print(e)
    finally:
        coord.request_stop()
        coord.join(threads)

print("finished")
