"""
Implementation of begin_training function, which builds:
* Optimisation options
* Gan Cost function
* Metrics and Summaries.
"""

# pip install tqdm colorama sklearn scipy Pillow matplotlib
# ^ needed to work
# --Disable execution timeout
import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import tflib as lib
from metrics import chi_square, L_norm, frechet_distance, cos_distance
import fid
from input_pipe import input_pipeline
import helpers
FLAGS = tf.app.flags.FLAGS
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Get rid of console garbage

N_GPUS = len(FLAGS.gpus_to_use.split(','))
# How many iterations to train the discriminator for
CRITIC_ITERS = FLAGS.critic_iters
LAMBDA = FLAGS.gradient_penalty
# Gradient penalty lambda hyperparameter
train_data_list = helpers.get_dataset_files()
# Batch size. Must be a multiple of N_GPUS
BATCH_SIZE = FLAGS.batch_size * N_GPUS
DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
print("Devices that will be used for training: ", DEVICES)
MODE = FLAGS.gan_version
EPOCH = FLAGS.images_per_file // BATCH_SIZE
OUTPUT_DIM = FLAGS.output_dim  # Number of pixels in each iamge


def begin_training(params):
    """
    Takes model name, Generator and Discriminator architectures as input,
    builds the rest of the graph.

    """
    model_name, Generator, Discriminator, epochs, restore = params
    fid_stats_file = "./tmp/"
    inception_path = "./tmp/"
    TRAIN_FOR_N_EPOCHS = epochs
    MODEL_NAME = model_name + "_" + FLAGS.dataset
    SUMMARY_DIR = 'summary/' + MODEL_NAME + "/"
    SAVE_DIR = "./saved_models/" + MODEL_NAME + "/"
    OUTPUT_DIR = './outputs/' + MODEL_NAME + "/"
    helpers.refresh_dirs(SUMMARY_DIR, OUTPUT_DIR, SAVE_DIR, restore)
    with tf.Graph().as_default():
        with tf.variable_scope('input'):
            all_real_data_conv = input_pipeline(
                train_data_list, batch_size=BATCH_SIZE)
            # Split data over multiple GPUs:
            split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
        global_step = tf.train.get_or_create_global_step()

        gen_cost, disc_cost, pre_real, pre_fake, gradient_penalty, real_data, fake_data, disc_fake, disc_real = split_and_setup_costs(
            Generator, Discriminator, split_real_data_conv)

        gen_train_op, disc_train_op, gen_learning_rate = setup_train_ops(
            gen_cost, disc_cost, global_step)

        performance_merged, distances_merged = add_summaries(gen_cost, disc_cost, fake_data, real_data,
                                                             gen_learning_rate, gradient_penalty, pre_real, pre_fake)

        saver = tf.train.Saver(max_to_keep=1)
        all_fixed_noise_samples = helpers.prepare_noise_samples(
            DEVICES, Generator)

        fid_stats_file += FLAGS.dataset + "_stats.npz"
        assert tf.gfile.Exists(
            fid_stats_file), "Can't find training set statistics for FID (%s)" % fid_stats_file
        f = np.load(fid_stats_file)
        mu_fid, sigma_fid = f['mu'][:], f['sigma'][:]
        f.close()
        inception_path = fid.check_or_download_inception(inception_path)
        fid.create_inception_graph(inception_path)

        # Create session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        if FLAGS.use_XLA:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as sess:
            # Restore variables if required
            ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
            if restore and ckpt and ckpt.model_checkpoint_path:
                print("Restoring variables...")
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Variables restored from:\n', ckpt.model_checkpoint_path)
            else:
                # Initialise all the variables
                print("Initialising variables")
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
                print('Variables initialised.')
            # Start input enqueue threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('Queue runners started.')
            real_im = sess.run([all_real_data_conv])[0][0][0][0:5]
            print("Real Image range sample: ", real_im)

            summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
            helpers.sample_dataset(sess, all_real_data_conv, OUTPUT_DIR)
            # Training loop
            try:
                ep_start = (global_step.eval(sess)) // EPOCH
                for epoch in tqdm(range(ep_start, TRAIN_FOR_N_EPOCHS), desc="Epochs passed"):
                    step = (global_step.eval(sess)) % EPOCH
                    for _ in tqdm(range(step, EPOCH), desc="Current epoch %i" % epoch, mininterval=0.5):
                        # train gen
                        _, step = sess.run([gen_train_op, global_step])
                        # Train discriminator
                        if (MODE == 'dcgan') or (MODE == 'lsgan'):
                            disc_iters = 1
                        else:
                            disc_iters = CRITIC_ITERS
                        for _ in range(disc_iters):
                            _disc_cost, _ = sess.run(
                                [disc_cost, disc_train_op])
                        if step % (128) == 0:
                            _, _, _, performance_summary, distances_summary = sess.run(
                                [gen_train_op, disc_cost, disc_train_op, performance_merged, distances_merged])
                            summary_writer.add_summary(
                                performance_summary, step)
                            summary_writer.add_summary(
                                distances_summary, step)

                        if step % (512) == 0:
                            saver.save(sess, SAVE_DIR, global_step=step)
                            helpers.generate_image(step, sess, OUTPUT_DIR,
                                                   all_fixed_noise_samples, Generator, summary_writer)
                            fid_score, IS_mean, IS_std, kid_score = fake_batch_stats(
                                sess, fake_data)
                            pre_real_out, pre_fake_out, fake_out, real_out = sess.run(
                                [pre_real, pre_fake, disc_fake, disc_real])
                            scalar_avg_fake = np.mean(fake_out)
                            scalar_sdev_fake = np.std(fake_out)
                            scalar_avg_real = np.mean(real_out)
                            scalar_sdev_real = np.std(real_out)

                            frechet_dist = frechet_distance(
                                pre_real_out, pre_fake_out)
                            kid_score = np.mean(kid_score)
                            inception_summary = tf.Summary()
                            inception_summary.value.add(
                                tag="distances/FD", simple_value=frechet_dist)
                            inception_summary.value.add(
                                tag="distances/FID", simple_value=fid_score)
                            inception_summary.value.add(
                                tag="distances/IS_mean", simple_value=IS_mean)
                            inception_summary.value.add(
                                tag="distances/IS_std", simple_value=IS_std)
                            inception_summary.value.add(
                                tag="distances/KID", simple_value=kid_score)
                            inception_summary.value.add(
                                tag="distances/scalar_mean_fake", simple_value=scalar_avg_fake)
                            inception_summary.value.add(
                                tag="distances/scalar_sdev_fake", simple_value=scalar_sdev_fake)
                            inception_summary.value.add(
                                tag="distances/scalar_mean_real", simple_value=scalar_avg_real)
                            inception_summary.value.add(
                                tag="distances/scalar_sdev_real", simple_value=scalar_sdev_real)
                            summary_writer.add_summary(inception_summary, step)
            except KeyboardInterrupt as e:
                print("Manual interrupt occurred.")
            except Exception as e:
                print(e)
            finally:
                coord.request_stop()
                coord.join(threads)
                print('Finished training.')
                saver.save(sess, SAVE_DIR, global_step=step)
                print("Model " + MODEL_NAME +
                      " saved in file: {} at step {}".format(SAVE_DIR, step))

def fake_batch_stats(sess, fake_data):
    """
    Makes a numpy array of fake images which can be fed into inception graph
    """
    batch_size = FLAGS.batch_size
    no_images = (5000 // batch_size) * batch_size
    temp_image_list = np.empty([no_images, FLAGS.height, FLAGS.height, 3])
    # Build a massive array of dataset images
    if FLAGS.data_format == "NCHW":
        out_shape = [batch_size, FLAGS.n_ch, FLAGS.height, FLAGS.height]
    else:
        out_shape = [batch_size, FLAGS.height, FLAGS.height, FLAGS.n_ch]

    if "mnist" in FLAGS.dataset:
        for i in range(no_images // batch_size):
            np_images = np.asarray(
                sess.run(tf.squeeze(tf.round(tf.reshape(fake_data, out_shape)))))
            temp_image_list[i * batch_size:(i + 1) *
                            batch_size, :, :, 0] = np_images
            temp_image_list[i * batch_size:(i + 1) *
                            batch_size, :, :, 1] = np_images
            temp_image_list[i * batch_size:(i + 1) *
                            batch_size, :, :, 2] = np_images
    else:
        for i in range(no_images // batch_size):
            np_images = np.asarray(
                sess.run(tf.squeeze(tf.round((tf.reshape(fake_data, out_shape) + 1) * 255))))
            temp_image_list[i * batch_size:(i + 1) *
                            batch_size, :, :, :] = np_images

    fid_score, IS_mean, IS_std, kid = fid.calculate_fid_IS_kid(
        temp_image_list, sess)

    return fid_score, IS_mean, IS_std, kid


def add_summaries(gen_cost, disc_cost, fake_data, real_data, gen_learning_rate, gradient_penalty, pre_real, pre_fake):
    print("Adding summaries")
    tf.summary.scalar('performance/G_loss', gen_cost,
                      collections=["performance"])
    tf.summary.scalar('performance/D_loss', disc_cost,
                      collections=["performance"])
    if FLAGS.gan_version == "wgan-gp":
        tf.summary.scalar("performance/gradient_penalty",
                          gradient_penalty, collections=["performance"])
    L2 = L_norm(pre_real, pre_fake, 2)
    cos = cos_distance(pre_real, pre_fake)
    tf.summary.scalar("distances/L2", L2, collections=["distances"])
    tf.summary.scalar("distances/cos", (1 - cos),
                      collections=["distances"])
    chi = chi_square(fake_data, real_data)
    tf.summary.scalar('distances/chi_square', chi,
                      collections=["distances"])

    tf.summary.scalar("performance/gen_learning_rate",
                      gen_learning_rate, collections=["performance"])

    performance_merged = tf.summary.merge_all(key="performance")
    distances_merged = tf.summary.merge_all(key="distances")
    return performance_merged, distances_merged


def setup_train_ops(gen_cost, disc_cost, global_step):
    gen_learning_rate = 1e-4
    if MODE == 'wgan':
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost,
                                                                              var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost,
                                                                               var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

        clip_ops = []
        for var in lib.params_with_name('Discriminator'):
            clip_bounds = [-.01, .01]
            clip_ops.append(tf.assign(var, tf.clip_by_value(
                var, clip_bounds[0], clip_bounds[1])))
        clip_disc_weights = tf.group(*clip_ops)

    elif MODE == 'wgan-gp':
        if FLAGS.decay_gen_lrate:
            gen_learning_rate = tf.train.exponential_decay(
                learning_rate=1e-4, global_step=global_step, decay_steps=1000, decay_rate=0.95)
        else:
            gen_learning_rate = 1e-4
        gen_train_op = tf.train.AdamOptimizer(gen_learning_rate, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                                                                var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True, global_step=global_step)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                                                  var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    elif MODE == 'dcgan':
        gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                                                      var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True, global_step=global_step)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                                                       var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    elif MODE == 'lsgan':
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost,
                                                                              var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True, global_step=global_step)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_cost,
                                                                               var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    else:
        raise Exception("Choose correct GAN version")
    return gen_train_op, disc_train_op, gen_learning_rate


def split_and_setup_costs(Generator, Discriminator, split_real_data_conv):
    gen_costs, disc_costs = [], []
    gradient_penalty = 0
    # Initialise the summaries and the log writers
    print("Splitting data over devices")
    for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
        with tf.device(device):
            real_data = tf.reshape(2 * ((tf.cast(real_data_conv, tf.float32) / 255.) - .5), [
                                   BATCH_SIZE // len(DEVICES), OUTPUT_DIM])
            fake_data = Generator(BATCH_SIZE // len(DEVICES))
            disc_real, pre_real = Discriminator(real_data)
            disc_fake, pre_fake = Discriminator(fake_data)

            if MODE == 'wgan':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(
                    disc_fake) - tf.reduce_mean(disc_real)

            elif MODE == 'wgan-gp':
                reduce_axis = [1]
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(
                    disc_fake) - tf.reduce_mean(disc_real)

                alpha = tf.random_uniform(
                    shape=[BATCH_SIZE // len(DEVICES), 1],
                    minval=0.,
                    maxval=1.
                )
                differences = fake_data - real_data
                interpolates = real_data + (alpha * differences)
                disc_interpolates, _ = Discriminator(interpolates)
                gradients = tf.gradients(
                    disc_interpolates, [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(
                    tf.square(gradients), axis=reduce_axis))
                gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
                disc_cost += LAMBDA * gradient_penalty

            elif MODE == 'dcgan':
                try:  # tf pre-1.0 (bottom) vs 1.0 (top)
                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                      labels=tf.ones_like(disc_fake)))
                    disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                       labels=tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                        labels=tf.ones_like(disc_real)))
                except Exception as e:
                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        disc_fake, tf.ones_like(disc_fake)))
                    disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        disc_fake, tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real,
                                                                                        tf.ones_like(disc_real)))
                disc_cost /= 2.

            elif MODE == 'lsgan':
                gen_cost = tf.reduce_mean((disc_fake - 1)**2)
                disc_cost = (tf.reduce_mean((disc_real - 1)**2) +
                             tf.reduce_mean((disc_fake - 0)**2)) / 2.

            else:
                raise Exception(
                    "You must choose a correct GAN version")
            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)
    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)
    return gen_cost, disc_cost, pre_real, pre_fake, gradient_penalty, real_data, fake_data, disc_fake, disc_real


if __name__ == '__main__':
    assert FLAGS.data_format in [
        "NCHW", "NHWC"], "Choose correct data format in flags.py [NCHW|NWHC]"
    assert FLAGS.gan_version in ["dcgan", "wgan", "wgan-gp",
                                 "lsgan"], "Choose one of the available gan versions in flags.py [dcgan|wgan|wgan-gp|lsgan]"
    assert FLAGS.dataset in [
        "mnist", "cifar10", "celebA"], "Choose one of the available datasets flags.py [mnist/lsun/cifar10/celebA]"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        FLAGS.gpus_to_use)  # Select which GPU to use
    tf.app.run()
