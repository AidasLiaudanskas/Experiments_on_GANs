'''
Outline of the test:
1. Implement 5-10 DCGAN-like Generator and Discriminator architectures with increasing complexity,
while maintaining them symmetric.
2. Train WGAN/WGAN-GP with those architectures
3. Load mismatched architectures and see the predictions of the discriminator -
how severe is the difference?

TODO:
1. Implement architectures - DONE
2. Implement training - DONE
3. Implement FID metric in order to compare different architectures? - will need it later anyways.
    3.1. Augment metrics module with IS and FID and KID
    - DONE

4. TEST ON GPU - Done
5. FIX Conv2D Transpose errors. - Done

'''
import os
from multiprocessing import Pool
import contextlib
import tensorflow as tf
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Get rid of console garbage
from flags import flags
from DCGANs import DCGAN
import tflib as lib
import numpy as np
from tqdm import tqdm
FLAGS = tf.app.flags.FLAGS

# fl = flags()
models_and_options = [
    ("DCGAN_64_64", 64, 64, 1, 1, "CelebA", False),
    ("DCGAN_32_32", 32, 32, 1, 1, "CelebA", False),
    ("DCGAN_16_16", 16, 16, 1, 1, "CelebA", False),
    ("DCGAN_8_8", 8, 8, 1, 1, "CelebA", False),
    ("DCGAN_4_4", 4, 4, 1, 1, "CelebA", False)
]
"""Make tuples of:
(model_name, G_dim, D_dim, G_ver, D_ver, datset_name, restore_initially)
"""

models_dict = {}
for m in models_and_options:
    models_dict[m[0]] = m

no_samples = 100


def test_function(params):
    DCG, gen, disc = params
    Generator = DCG.DCGANG_1
    Discriminator = DCG.DCGAND_1
    BATCH_SIZE = FLAGS.batch_size
    with tf.Graph().as_default() as graph:
        fake_data = Generator(BATCH_SIZE)
        disc_fake, pre_fake = Discriminator(fake_data)
        gen_vars = lib.params_with_name('Generator')
        gen_saver = tf.train.Saver(gen_vars)
        disc_vars = lib.params_with_name("Discriminator")
        disc_saver = tf.train.Saver(disc_vars)
        ckpt_gen = tf.train.get_checkpoint_state(
            "./saved_models/" + gen + "/")
        ckpt_disc = tf.train.get_checkpoint_state(
            "./saved_models/" + disc + "/")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if ckpt_gen and ckpt_gen.model_checkpoint_path:
                print("Restoring generator...", gen)
                gen_saver.restore(sess, ckpt_gen.model_checkpoint_path)
            else:
                print("Failed to load Generator")
            if ckpt_disc and ckpt_disc.model_checkpoint_path:
                print("Restoring discriminator...", disc)
                disc_saver.restore(
                    sess, ckpt_disc.model_checkpoint_path)
                pred_arr = np.empty([no_samples, BATCH_SIZE])
                for i in tqdm(range(no_samples + 1)):
                    predictions = sess.run([disc_fake])
                    pred_arr[i - 1, :] = predictions[0]
                overall_mean = np.mean(pred_arr)
                overall_std = np.std(pred_arr)
                batch_means = np.mean(np.mean(pred_arr, axis=1))
                batch_stds = np.std(np.std(pred_arr, axis=1))
                return pred_arr
            else:
                print("Failed to load Discriminator")


def evaluate():
    """
    Should be able to mix and match various versions of GANs with this function. Steps:
    1. List all models available in the save_dir
    2. Double for loop:
     2.1 For each Generator
     2.2 For load each discriminator and
    3. Run 100k samples and see the discriminator output.
    """
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
    print("Save files found: ", save_files)
    print("Depths parsed: ", indexes)
    l = len(save_files)
    results_tensor = np.empty([l * l, no_samples, FLAGS.batch_size])
    DCG = DCGAN()
    num_pool_workers = 1
    i = 0
    for j, gen in enumerate(save_files):
        for k, disc in enumerate(save_files):
            DCG.set_G_dim(indexes[j])
            print("G_dim set to ", indexes[j])
            DCG.set_D_dim(indexes[k])
            print("D_dim set to ", indexes[k])
            param_tuple = (DCG, gen, disc)
            with contextlib.closing(Pool(num_pool_workers)) as po:
                pool_results = po.map_async(
                    test_function, (param_tuple for _ in range(1)))
                results_list = pool_results.get()
                results_tensor[i] = results_list[0]
                i += 1
    print("Evaluation finished")
    output_path = './evaluation_stats.npz'
    np.savez_compressed(
        output_path, results_tensor=results_tensor, save_files=save_files)
    print("Output saved")

"""
Brainstorming on how to implement the evaluation:

* Two graphs with two sessions, reset each on new architecture?
Unavoidably O(N^2) anyways.

* Two graphs in one session? - Potentially faster, don't really see how it'd work.

TODO: Get rid of this no variables to restore error!!! - Done
"""


if __name__ == '__main__':
    evaluate()
