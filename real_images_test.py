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
import helpers
from input_pipe import input_pipeline
FLAGS = tf.app.flags.FLAGS

fl = flags()

no_samples = FLAGS.images_per_file // FLAGS.batch_size

def test_function(params):
    DCG, disc = params
    Discriminator = DCG.DCGAND_1
    BATCH_SIZE = FLAGS.batch_size
    with tf.Graph().as_default() as graph:
        train_data_list = helpers.get_dataset_files()
        real_data = input_pipeline(train_data_list, batch_size=BATCH_SIZE)
        # Normalize -1 to 1
        real_data = 2 * ((tf.cast(real_data, tf.float32) / 255.) - .5)
        disc_real, _ = Discriminator(real_data)
        disc_vars = lib.params_with_name("Discriminator")
        disc_saver = tf.train.Saver(disc_vars)
        ckpt_disc = tf.train.get_checkpoint_state(
            "./saved_models/" + disc + "/")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('Queue runners started.')
            real_im = sess.run([real_data])[0][0][0][0:5]
            print("Real Image range sample: ", real_im)
            try:
                if ckpt_disc and ckpt_disc.model_checkpoint_path:
                    print("Restoring discriminator...", disc)
                    disc_saver.restore(
                        sess, ckpt_disc.model_checkpoint_path)
                    pred_arr = np.empty([no_samples, BATCH_SIZE])
                    for i in tqdm(range(no_samples + 1)):
                        predictions = sess.run([disc_real])
                        pred_arr[i - 1, :] = predictions[0]
                    return pred_arr
                else:
                    print("Failed to load Discriminator")
            except KeyboardInterrupt as e:
                print("Manual interrupt occurred.")
            finally:
                coord.request_stop()
                coord.join(threads)
                print('Finished inference.')


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
    save_files = [x for x in save_files if (
        x.split("_")[1] == x.split("_")[2])]
    indexes = [int(x.split("_")[1]) for x in save_files]
    save_files = [x for _, x in sorted(zip(indexes, save_files))]
    indexes = sorted(indexes)
    print("Save files found: ", save_files)
    # TODO: Filter save files to only load symmetrically trained ones... i.e. indexes mus be equal.
    print("Depths parsed: ", indexes)
    l = len(save_files)
    results_tensor = np.empty([l, no_samples, FLAGS.batch_size])
    DCG = DCGAN()
    num_pool_workers = 1
    i = 0
    for k, disc in enumerate(tqdm(save_files)):
        DCG.set_D_dim(indexes[k])
        print("D_dim set to ", indexes[k])
        param_tuple = (DCG, disc)
        with contextlib.closing(Pool(num_pool_workers)) as po:
            pool_results = po.map_async(
                test_function, (param_tuple for _ in range(1)))
            results_list = pool_results.get()
            results_tensor[i] = results_list[0]
            i += 1
    print("Evaluation finished")
    output_path = './real_evaluation_stats.npz'
    np.savez_compressed(
        output_path, results_tensor=results_tensor, save_files=save_files)
    print("Output saved")


if __name__ == '__main__':
    evaluate()
