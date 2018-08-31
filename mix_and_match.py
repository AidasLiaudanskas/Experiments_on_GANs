'''
TL;DR:
Specify desired hiperparameters in models_and_options tuple list. Run this file.
Watch the training converge/diverge.


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
from flags import flags
FLAGS = tf.app.flags.FLAGS
from DCGANs import DCGAN
import tflib as lib
import numpy as np
from begin_gan_training import begin_training
from mix_and_match_test import evaluate
from tqdm import tqdm
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Get rid of console garbage
fl = flags()

models_and_options = [
    ("DCGAN_4_4", 4, 4, 1, 1, "CelebA", False),
    # ("DCGAN_8_8", 8, 8, 1, 1, "CelebA", False),
    # ("DCGAN_6_6", 6, 6, 1, 1, "CelebA", False),
    # ("DCGAN_12_12", 12, 12, 1, 1, "CelebA", False),
    # ("DCGAN_16_4", 16, 4, 1, 1, "CelebA", False),
    # ("DCGAN_16_8", 16, 8, 1, 1, "CelebA", False),
    # ("DCGAN_16_12", 16, 12, 1, 1, "CelebA", False)
    # ("DCGAN_16_16", 16, 16, 1, 1, "CelebA", False),
    # ("DCGAN_20_20", 20, 20, 1, 1, "CelebA", False),
    # ("DCGAN_24_24", 24, 24, 1, 1, "CelebA", False),
    # ("DCGAN_32_32", 32, 32, 1, 1, "CelebA", False),
    # ("DCGAN_64_16", 64, 16, 1, 1, "CelebA", False),
    # ("DCGAN_20_20", 20, 20, 1, 1, "CelebA", True),
    # ("DCGAN_22_22", 22, 22, 1, 1, "CelebA", False),
    # ("DCGAN_26_26", 26, 26, 1, 1, "CelebA", False),
    # ("DCGAN_28_28", 28, 28, 1, 1, "CelebA", True),
    # ("DCGAN_64_32", 64, 32, 1, 1, "CelebA", False),
    # ("DCGAN_64_48", 64, 48, 1, 1, "CelebA", False),
    # ("DCGAN_32_8", 32, 8, 1, 1, "CelebA", True),
    # ("DCGAN_32_16", 32, 16, 1, 1, "CelebA", True),
    # ("DCGAN_32_24", 32, 24, 1, 1, "CelebA", True),
    # ("DCGAN_32_40", 32, 40, 1, 1, "CelebA", True),
    # ("DCGAN_32_48", 32, 48, 1, 1, "CelebA", True),
    # ("DCGAN_32_56", 32, 56, 1, 1, "CelebA", True),
    # ("DCGAN_32_64", 32, 64, 1, 1, "CelebA", True),
    # ("DCGAN_40_40", 40, 40, 1, 1, "CelebA", False),
    # ("DCGAN_48_48", 48, 48, 1, 1, "CelebA", False),
    # ("DCGAN_56_56", 56, 56, 1, 1, "CelebA", False),
    # ("DCGAN_64_64", 64, 64, 1, 1, "CelebA", True),
    # ("DCGAN_72_72", 72, 72, 1, 1, "CelebA", True),
    # ("DCGAN_80_80", 80, 80, 1, 1, "CelebA", True),
    # ("DCGAN_100_100", 100, 100, 1, 1, "CelebA", True),
    ("DCGAN_128_128", 128, 128, 1, 1, "CelebA", True),
]

"""Make tuples of:
(model_name, G_dim, D_dim, G_ver, D_ver, datset_name, restore_initially).
dataset_name here is just for save_file naming purposes, to change the dataset used
for training, have to change it in flags.py
"""


def launch_managed_training():
    """
    Function which trains many models in series
    """
    epochs = 10
    DCG = DCGAN()
    num_pool_workers = 1  # can be bigger than 1, to enable parallel execution
    for entry in models_and_options:
        print("Starting training of model: \n", entry)
        DCG.set_G_dim(entry[1])
        DCG.set_D_dim(entry[2])
        if entry[3] == 1:
            Generator = DCG.DCGANG_1
        else:
            Generator = DCG.DCGANG_2
        if entry[4] == 1:
            Discriminator = DCG.DCGAND_1
        else:
            Discriminator = DCG.DCGAND_2
        param_tuple = (entry[0], Generator, Discriminator, epochs, entry[6])
        with contextlib.closing(Pool(num_pool_workers)) as po:
            pool_results = po.map_async(
                begin_training, (param_tuple for _ in range(1)))
            results_list = pool_results.get()
    # This ensures that the processes get closed once they are done
    return 0


if __name__ == '__main__':
    launch_managed_training()
