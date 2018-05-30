import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# a = flags()
# This file defines default values. They can be altered here or in command line
# image format is by default NCHW

# works in Python 2 & 3, according to https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python


class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                _Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(_Singleton('SingletonMeta', (object,), {})):
    pass

# Need a singleton so that this class can only be initialised once.


class flags(Singleton):
    def __init__(self):
        dataset = "celebA_64x64"
        batch_size = 256
        if dataset == "mnist_32x32":
            images_per_file = 60000
            image_height = 32
            n_ch = 1
        if dataset == "lsun_bedrooms_64x64":
            images_per_file = 500000
            image_height = 64
            n_ch = 3
        if dataset == "mnist":
            images_per_file = 70000
            image_height = 28
            n_ch = 1
        if dataset == "celebA_64x64":
            images_per_file = 202600
            image_height = 64
            n_ch = 3
        if dataset == "cifar10":
            images_per_file = 60000
            image_height = 32
            n_ch = 3
        output_dim = image_height * image_height * n_ch
        self.define_flags(dataset, n_ch, images_per_file,
                          output_dim, image_height, batch_size)
        print("Flags are initialized")

    def define_flags(self, dataset, n_ch, images_per_file, output_dim, image_height, batch_size):
        ############
        # Model_specific parameters
        ############
        # tf.app.flags.DEFINE_string('run', None, "Which operation to run. [train|inference]")
        tf.app.flags.DEFINE_integer(
            'model_dim', 64, "Dimensionality of the model")
        tf.app.flags.DEFINE_string(
            'dataset', dataset, "mnist/lsun/cifar10/celebA")
        tf.app.flags.DEFINE_string(
            'architecture', 'dcgan', "dcgan/ResNet/etc")
        tf.app.flags.DEFINE_integer(
            'critic_iters', 5, "For each Generator iteration, train Discriminator for n iterations")
        tf.app.flags.DEFINE_integer(
            'gradient_penalty', 10, "Gradient penalty term")
        tf.app.flags.DEFINE_integer(
            'n_ch', n_ch, "No of channels in the input image. RGB = 3, B&W = 1")
        tf.app.flags.DEFINE_integer(
            'output_dim', output_dim, "No of pixels in the output image")
        tf.app.flags.DEFINE_integer(
            'height', image_height, "Image side size")
        tf.app.flags.DEFINE_string(
            "gan_version", "wgan-gp", "Version of GAN to use")
        # Choose above from dcgan, wgan, wgan-gp, lsgan
        tf.app.flags.DEFINE_boolean(
            'restore', False, "Do you want to restore a previous model (if it exists)")
        # tf.app.flags.DEFINE_integer('no_of_gpus', 1, "How many GPUs to use")
        tf.app.flags.DEFINE_string(
            'gpus_to_use', '0', "Specify which GPUs to use in format: 0,1,2,3")

        ##########################
        # Training parameters
        ###########################
        tf.app.flags.DEFINE_integer(
            'nb_epoch', 4, "Number of epochs to train for")
        tf.app.flags.DEFINE_integer(
            'batch_size', batch_size, "Number of samples per batch. Scales with no_of_gpus")
        # tf.app.flags.DEFINE_integer(
        #     'nb_batch_per_epoch', 50, "Number of batches per epoch")
        tf.app.flags.DEFINE_float('learning_rate', 2E-4,
                                  "Learning rate used for AdamOptimizer")
        tf.app.flags.DEFINE_integer(
            'noise_dim', 128, "Noise dimension for GAN generation")
        tf.app.flags.DEFINE_integer(
            'random_seed', 0, "Seed used to initialize rng.")
        tf.app.flags.DEFINE_boolean('decay_gen_lrate', False,
                                    "Whether to Decay the learning rate of Generator. \
                                    A way to make it less powerful. Seemed to work with WGAN-GP")

        ############################################
        # General tensorflow parameters parameters
        #############################################
        tf.app.flags.DEFINE_boolean(
            'use_XLA', True, "Whether to use XLA compiler.")
        tf.app.flags.DEFINE_integer(
            'num_threads', 4, "Number of threads to fetch the data")
        tf.app.flags.DEFINE_float('capacity_factor', 4,
                                  "Number of batches to store in queue")

        ##########
        # Datasets
        ##########
        # tf.app.flags.DEFINE_string('data_format', "NHWC", "Tensorflow image data format.")
        # CPU support and NHWC hasn't been thoroughly tested, use GPU mode for reliable results
        tf.app.flags.DEFINE_string('data_format', "NHWC",
                                   "Tensorflow image data format: NHWC or NCHW. CPU only supports NHWC")
        # tf.app.flags.DEFINE_boolean(
        #     'use_GPU', True, "Whether to use GPU.")
        tf.app.flags.DEFINE_string(
            'dataset_path', "../data", "Path to where the tfrecords files are")
        tf.app.flags.DEFINE_integer('images_per_file', images_per_file,
                                    "Number of pictures per tfrecords file, so we'd know how much an epoch is")

        # tf.app.flags.DEFINE_integer('channels', 3, "Number of channels")
        tf.app.flags.DEFINE_float('central_fraction', 0.8,
                                  "Central crop as a fraction of total image")

        ##############
        # Directories
        ##############
        tf.app.flags.DEFINE_string(
            'model_dir', './saved_models', "Output folder where checkpoints are dumped.")
        tf.app.flags.DEFINE_string(
            'log_dir', './logs', "Logs for tensorboard.")
        tf.app.flags.DEFINE_string(
            'fig_dir', './figures', "Where to save figures.")


a = flags()
