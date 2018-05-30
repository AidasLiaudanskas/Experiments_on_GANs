'''
Outline of the test:
1. Generate one image with a G.
2. Load up the LIME explainer with D,
2.2. Map D output to two classes and limit with sigmoid.


The ultimate goal is to make outputs like that:
for every G:
    for every D:
        generate 20 images with explanations.

also do the same with real images:
for every D:
    show 20 images with explanations


'''

import os
from multiprocessing import Pool
import argparse
import contextlib
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray  # since the code wants color images
# to make a nice montage of the images
from skimage.util.montage import montage2d
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Get rid of console garbage
from flags import flags
FLAGS = tf.app.flags.FLAGS
# Cheat to make flags work with notebooks:
tf.app.flags.DEFINE_string('f', '', 'kernel')
from DCGANs import DCGAN
import tflib as lib
from tqdm import tqdm
import lime
from lime import lime_image
import time
from scipy.special import expit
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries


"""
Complications:
Discriminator expects a batch size of 256, which is hardcoded. If I want to do inference on
a single image I'll have to augment the batch to 256 with potentially zeros at the end.
Slowing down everything horrendously -.-

Solved.

"""


no_samples = 16
# I.e. number of images to generate with explanations ^


no_perturbed_images = 2048
# Change back to 2048 after prototyping is done. ^

noise = np.random.normal(size=[no_samples, 128])

stats_file = "./mean_offsets.npz"
f = np.load(stats_file)
f.keys()
means_matrix = f["means_matrix"]

# normalize_by_mean = True


def explain(params=None):
    DCG, gen, disc, g_index, d_index, normalize_by_mean = params
    # DCG = DCGAN()
    # DCG.set_G_dim(64)
    # DCG.set_D_dim(64)
    # gen = disc = "DCGAN_64_64_celebA_64x64"
    Generator = DCG.DCGANG_1
    Discriminator = DCG.DCGAND_1
    BATCH_SIZE = FLAGS.batch_size
    with tf.Graph().as_default() as graph:
        noise_tf = tf.convert_to_tensor(noise, dtype=tf.float32)
        fake_data = Generator(BATCH_SIZE)
        # print("Fake_data shape: ", fake_data.shape)
        disc_fake, pre_fake = Discriminator(fake_data)
        # print("disc_fake shape: ", disc_fake.shape)
        gen_vars = lib.params_with_name('Generator')
        gen_saver = tf.train.Saver(gen_vars)
        disc_vars = lib.params_with_name("Discriminator")
        disc_saver = tf.train.Saver(disc_vars)
        ckpt_gen = tf.train.get_checkpoint_state(
            "./saved_models/" + gen + "/")
        ckpt_disc = tf.train.get_checkpoint_state(
            "./saved_models/" + disc + "/")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if ckpt_gen and ckpt_gen.model_checkpoint_path:
                # print("Restoring generator...", gen)
                gen_saver.restore(sess, ckpt_gen.model_checkpoint_path)
            else:
                print("Failed to load Generator", gen)
            if ckpt_disc and ckpt_disc.model_checkpoint_path:
                # print("Restoring discriminator...", disc)
                disc_saver.restore(
                    sess, ckpt_disc.model_checkpoint_path)

                def disc_prediction(image):
                    # make fake batch:
                    # Transform to -1 to 1:
                    if np.max(image) > 1.1 or np.min(image) > 0.0:
                        image = (image.astype(np.float32) * 2.0 / 255.0) - 1.0
                    if len(image.shape) == 4:
                        no_ims = image.shape[0]
                    else:
                        no_ims = 1
                    images_batch = np.zeros(
                        [256, 64, 64, 3]).astype(np.float32)
                    images_batch[0:no_ims] = image
                    prediction, _ = sess.run([Discriminator(images_batch)])[0]
                    # Need to map input from [-inf, + inf] to [-1, +1]
                    pred_array = np.zeros((no_ims, 2))
                    for i, x in enumerate(prediction[:no_ims]):
                        if normalize_by_mean:
                            bias = means_matrix[g_index][d_index]
                            pred_array[i, 1] = expit(x-bias)
                            pred_array[i, 0] = 1 - pred_array[i, 1]
                        else:
                            pred_array[i, 1] = expit(x)
                            pred_array[i, 0] = 1 - pred_array[i, 1]
                        # 1 == REAL; 0 == FAKE
                    return pred_array
                # images_to_explain = np.zeros([no_samples, 64, 64, 3])
                images_to_explain = sess.run(
                    [Generator(no_samples, noise=noise_tf)])[0]
                images_to_explain = (images_to_explain + 1.0) * 255.0 / 2.0
                images_to_explain = images_to_explain.astype(np.uint8)
                images_to_explain = np.reshape(
                    images_to_explain, [no_samples, 64, 64, 3])
                explanations = []
                explainer = lime_image.LimeImageExplainer(verbose=False)
                segmenter = SegmentationAlgorithm(
                    'slic', n_segments=100, compactness=1, sigma=1)
                for image_to_explain in tqdm(images_to_explain):
                    # tmp = time.time()
                    explanation = explainer.explain_instance(image_to_explain,
                                                             classifier_fn=disc_prediction, batch_size=256,
                                                             top_labels=2, hide_color=None, num_samples=no_perturbed_images,
                                                             segmentation_fn=segmenter)
                    # print(time.time() - tmp)
                    explanations.append(explanation)
                make_figures(images_to_explain, explanations,
                             DCG.get_G_dim(), DCG.get_D_dim(), normalize_by_mean)
            else:
                print("Failed to load Discriminator", disc)


def make_figures(images_to_explain, explanations, G_dim, D_dim, normalize_by_mean=True, n=no_samples):
    """
    visualizes images on a n x n grid
    shows explanations on the side.
    """
    assert len(images_to_explain) == n, "Incorrect number of images passed"
    assert n % 2 == 0, "n must be even"
    fig, ax = plt.subplots(nrows=4, ncols=n, sharex=True,
                           sharey=True, figsize=(16, 8))
    for l, image in enumerate(images_to_explain):
        plt.subplot(4, n // 2, l + 1)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4, n // 2, l + 1 + n)
        temp, mask = explanations[l].get_image_and_mask(
            1, positive_only=False, num_features=50, hide_rest=False)
        plt.imshow(mark_boundaries(temp, mask))
        plt.xticks([])
        plt.yticks([])
    # print("Figures ready")
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    # plt.suptitle('LIME Explanations for G.dim = {}, D.dim = {}'.format(G_dim, D_dim))
    plt.tight_layout()
    save_path = "./outputs/LIME/"
    save_name = "lime_G_{}_D_{}.png".format(G_dim, D_dim)
    if normalize_by_mean:
        save_name = "norm_lime_G_{}_D_{}.png".format(G_dim, D_dim)
    # plt.show()
    plt.savefig(save_path + save_name, dpi=None, format="png",)


def evaluate(min=None, max=None):
    """
    Should be able to mix and match various versions of GANs with this function. Steps:
    1. List all models available in the save_dir
    2. Double for loop:
     2.1 For each Generator
     2.2 For load each discriminator and
    3. Run 100k samples and see the discriminator output.
    """
    # os.chdir("/home/aidas/GAN_Experiments/progressive_test")
    current_dir = os.getcwd()
    print("Current_dir = ", current_dir)
    model_dir = "./saved_models"
    save_files = os.listdir(model_dir)
    # Filter only symmetric versions:
    save_files = [x for x in save_files if (
        x.split("_")[1] == x.split("_")[2])]
    indexes = [int(x.split("_")[1]) for x in save_files]
    save_files = [x for _, x in sorted(zip(indexes, save_files))]
    indexes = sorted(indexes)
    # save_files = save_files[-10:]
    # indexes = indexes[-10:]
    print("Save files found: ", save_files)
    print("Depths parsed: ", indexes)
    l = len(save_files)
    if not min:
        min = 0
    if not max:
        max = l
    DCG = DCGAN()
    i = 0
    for normalize_by_mean in [True, False]:
        for j, gen in enumerate(tqdm(save_files[min:max])):
            for k, disc in enumerate(tqdm(save_files)):
                DCG.set_G_dim(indexes[min+j])
                # print("G_dim set to ", indexes[j])
                DCG.set_D_dim(indexes[k])
                # print("D_dim set to ", indexes[k])
                # test_function(DCG, gen, disc)
                param_tuple = (DCG, gen, disc, j, k, normalize_by_mean)
                with contextlib.closing(Pool(1)) as po:
                    pool_results = po.map_async(
                        explain, (param_tuple,))
                    results_list = pool_results.get()
    print("Evaluation finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add min max generator indexes to process')
    # Allow parallelization of the script

    parser.add_argument('opt_min', type=int, nargs='?',
                        help='An optional min index value for generators evaluation')
    parser.add_argument('opt_max', type=int, nargs='?',
                        help='An optional max index value for generators evaluation')

    args = parser.parse_args()
    min, max = args.opt_min, args.opt_max
    print("Starting index = {}, ending index = {}".format(min, max))

    evaluate(min, max)
