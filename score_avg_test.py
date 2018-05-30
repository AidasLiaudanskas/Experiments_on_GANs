'''
Tests the mean of the output for each combination of G and D for real, fake and mixed dataself.
The mean output of mixed data is the mean of real and fake as expected.
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
import helpers
from input_pipe import input_pipeline
from tqdm import tqdm
FLAGS = tf.app.flags.FLAGS

no_samples = 100


def test_function(params):
    DCG, gen, disc = params
    Generator = DCG.DCGANG_1
    Discriminator = DCG.DCGAND_1
    BATCH_SIZE = FLAGS.batch_size
    with tf.Graph().as_default() as graph:
        fake_data = Generator(BATCH_SIZE)
        # print("Fake_data shape: ", fake_data.shape)
        disc_fake, _ = Discriminator(fake_data)

        train_data_list = helpers.get_dataset_files()
        real_data = input_pipeline(train_data_list, batch_size=BATCH_SIZE)
        # Normalize -1 to 1
        real_data = 2 * ((tf.cast(real_data, tf.float32) / 255.) - .5)
        # print("Fake_data shape: ", fake_data.shape)
        disc_real, _ = Discriminator(real_data)

        # print("disc_fake shape: ", disc_fake.shape)
        gen_vars = lib.params_with_name('Generator')
        gen_saver = tf.train.Saver(gen_vars)
        disc_vars = lib.params_with_name("Discriminator")
        disc_saver = tf.train.Saver(disc_vars)
        ckpt_gen = tf.train.get_checkpoint_state(
            "./saved_models/" + gen + "/")
        ckpt_disc = tf.train.get_checkpoint_state(
            "./saved_models/" + disc + "/")
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.66)
        config = tf.ConfigProto(allow_soft_placement=True,
                                # gpu_options=gpu_options,
                                )
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            try:
                sess.run(tf.global_variables_initializer())
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                # print('Queue runners started.')
                if ckpt_gen and ckpt_gen.model_checkpoint_path:
                    # print("Restoring generator...", gen)
                    # TODO: Not sure which one to use
                    gen_saver.restore(sess, ckpt_gen.model_checkpoint_path)
                else:
                    print("Failed to load Generator")
                if ckpt_disc and ckpt_disc.model_checkpoint_path:
                    # print("Restoring discriminator...", disc)
                    disc_saver.restore(
                        sess, ckpt_disc.model_checkpoint_path)
                    pred_arr = np.empty([3, no_samples, BATCH_SIZE],dtype=np.float32)
                    # print(rl.shape, fk.shape)
                    # print(half_real)
                    for i in tqdm(range(no_samples)):
                        rl, fk = sess.run([real_data, fake_data])
                        fk = fk.reshape([256, 64, 64, 3])
                        half_real = np.zeros([BATCH_SIZE, 64, 64, 3], dtype=np.float32)
                        half_real[:BATCH_SIZE//2] = rl[BATCH_SIZE//2 :]
                        half_real[BATCH_SIZE//2:] = fk[BATCH_SIZE//2 :]
                        predictions = sess.run([disc_fake])[0]
                        pred_arr[0, i, :] = predictions
                        # print(predictions)
                        predictions = sess.run([disc_real])[0]
                        pred_arr[1, i, :] = predictions
                        # print(predictions)
                        predictions = sess.run([Discriminator(half_real)])[0][0]
                        pred_arr[2, i, :] = predictions
                        # print(predictions)
                    fake_mean = np.mean(pred_arr[0])
                    real_mean = np.mean(pred_arr[1])
                    half_mean = np.mean(pred_arr[2])
                    fake_std = np.std(pred_arr[0])
                    real_std = np.std(pred_arr[1])
                    half_std = np.std(pred_arr[2])
                    coord.request_stop()
                    coord.join(threads)
                    return real_mean, fake_mean, half_mean, fake_std, real_std, half_std
                else:
                    print("Failed to load Discriminator")
            except Exception as e:
                print(e)
            finally:
                coord.request_stop()
                coord.join(threads)
                # print('Finished inference.')



def evaluate():
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
    save_files = [x for x in save_files if (x.split("_")[1] == x.split("_")[2])]
    indexes = [int(x.split("_")[1]) for x in save_files]
    save_files = [x for _,x in sorted(zip(indexes,save_files))]
    indexes = sorted(indexes)
    print("Save files found: ", save_files)
    # TODO: Filter save files to only load symmetrically trained ones... i.e. indexes mus be equal.
    print("Depths parsed: ", indexes)
    l = len(save_files)
    results_tensor = np.empty([l * l, 6])
    # I.e. for each of l architectures, show prediction mean for real, fake, mixed
    DCG = DCGAN()
    num_pool_workers = 1
    i = 0
    for j, gen in enumerate(tqdm(save_files)):
        for k, disc in enumerate(tqdm(save_files)):
            DCG.set_G_dim(indexes[j])
            # print("G_dim set to ", indexes[j])
            DCG.set_D_dim(indexes[k])
            # print("D_dim set to ", indexes[k])
            param_tuple = (DCG, gen, disc)
            with contextlib.closing(Pool(num_pool_workers)) as po:
                pool_results = po.map_async(
                    test_function, (param_tuple for _ in range(1)))
                results_list = pool_results.get()
                results_tensor[i] = results_list[0]
                i += 1
    print("Evaluation finished")
    output_path = './real_fake_half.npz'
    results_tensor.reshape([l,l,6])
    np.savez_compressed(output_path, results_tensor=results_tensor, save_files=save_files)
    print("Output saved")

if __name__ == '__main__':
    # flags.define_flags()
    # fl = flags()
    # launch_managed_training()
    evaluate()
