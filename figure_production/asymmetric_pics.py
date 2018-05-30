"""
The goal is to make a batch of pictures from different generators from different asymmetricly trained architectures

use the standard picture producing function

Construct a batch of 8 different Generators
32 images from each generator

"""
import os
from multiprocessing import Pool
import contextlib
import tensorflow as tf
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Get rid of console garbage
from flags import flags
tf.app.flags.DEFINE_string('f', '', 'kernel')
from DCGANs import DCGAN
import tflib as lib
import numpy as np
from tqdm import tqdm
FLAGS = tf.app.flags.FLAGS
import time
n_samples = 64
test_for_G = 64
noise = np.random.normal(size=[n_samples, 128])
# noise.shape
def test_function(params):
    DCG, gen = params
    Generator = DCG.DCGANG_1
    BATCH_SIZE = FLAGS.batch_size
    # print("Hi there")
    with tf.Graph().as_default() as graph:
        noise_tf = tf.convert_to_tensor(noise, dtype = tf.float32)
        fake_data = Generator(noise.shape[0], noise=noise_tf)
        print("Fake_data shape: ", fake_data.shape)
        # print("disc_fake shape: ", disc_fake.shape)
        gen_vars = lib.params_with_name('Generator')
        gen_saver = tf.train.Saver(gen_vars)
        ckpt_gen = tf.train.get_checkpoint_state(
            "./saved_models/" + gen + "/")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if ckpt_gen and ckpt_gen.model_checkpoint_path:
                print("Restoring generator...", gen)
                gen_saver.restore(sess, ckpt_gen.model_checkpoint_path)
                fake_images = sess.run([fake_data])[0]
                # return only 16
                fake_images = fake_images.reshape([noise.shape[0],64,64,3])
                print("fake_images shape: ", np.shape(fake_images))
                return fake_images
            else:
                print("Failed to load Generator")


def generate_png(fake_data, gen_dim=None, name=None, output_dir="./outputs/assymetric_imgs"):
    height = FLAGS.height
    no_imgs = fake_data.shape[0]
    if FLAGS.data_format == "NCHW":
        output_shape = [no_imgs, FLAGS.n_ch, height, height]
    else:
        output_shape = [no_imgs, height, height, FLAGS.n_ch]
    samples = fake_data
    samples = ((samples + 1.) * (255.99 / 2)).astype('int32')
    samples = np.reshape(samples, output_shape)
    print("Samples shape: ", samples.shape)
    if FLAGS.data_format == "NHWC" and samples.shape[3] in [1, 3]:
        # NHWC --> NCHW
        samples = np.transpose(samples, [0, 3, 1, 2])
    rnd = int(time.time()%1000)
    if name == None:
        name = output_dir + '/G_{}_W_D_16-64_{}.png'.format(test_for_G, rnd)
    lib.save_images.save_images(samples, name)
    print("image " + name + " saved")


def evaluate():
    current_dir = os.getcwd()
    print("Current_dir = ", current_dir)
    model_dir = "./saved_models"
    save_files = os.listdir(model_dir)
    # Filter only asymmetric versions:
    save_files = [x for x in save_files if (int(x.split("_")[1]) == test_for_G)]
    indexes = [int(x.split("_")[2]) for x in save_files]
    save_files = [x for _,x in sorted(zip(indexes,save_files))]
    indexes = sorted(indexes)
    print("Save files found: ", save_files)
    # TODO: Filter save files to only load symmetrically trained ones... i.e. indexes mus be equal.
    print("Depths parsed: ", indexes)
    l = len(save_files)
    image_batch = np.zeros([256, 64, 64, 3])
    DCG = DCGAN()
    num_pool_workers = 1
    i = 0
    noise = tf.random_normal([n_samples, 128])
    # print("Noise: ", noise)
    for j, gen in enumerate(save_files):
        DCG.set_G_dim(test_for_G)
        print("G_dim set to ", test_for_G)
        # test_function(DCG, gen, disc)
        param_tuple = (DCG, gen)
        with contextlib.closing(Pool(num_pool_workers)) as po:
            pool_results = po.map_async(
                test_function, (param_tuple,))
            results_list = pool_results.get()
            image_batch[i*n_samples:n_samples*(i+1)] = results_list[0]
            i += 1
    # print(image_batch)
    generate_png(image_batch)
    print("Output saved")



if __name__ == '__main__':
    evaluate()
