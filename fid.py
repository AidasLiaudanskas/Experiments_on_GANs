#!/usr/bin/env python3
''' Calculates the Frechet Inception Distance (FID) to evalulate GANs.

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

See --help to see further details.
'''

from __future__ import absolute_import, division, print_function
import pathlib
import os
import sys
import warnings
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from scipy import linalg
from tqdm import tqdm
from KID_score import polynomial_mmd_averages
import flags
FLAGS = tf.app.flags.FLAGS



class InvalidFIDException(Exception):
    pass


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')
#-------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    # print("Returning layers")
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o._shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name(
        "FID_Inception_Net/softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)
    return pool3, softmax
#-------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=50, verbose=True, splits=10):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    # print("Entering get_activations")
    inception_layer, softmax_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size
    pred_inc_arr = np.empty((n_used_imgs, 2048))
    pred_soft_arr = []
    for i in tqdm(range(n_batches)):
        if verbose:
            print("\rPropagating batch %d/%d" %
                  (i + 1, n_batches), end="", flush=True)
        start = i * batch_size
        end = start + batch_size
        batch = images[start:end]
        pred, pred_soft = sess.run([inception_layer, softmax_layer], {
            'FID_Inception_Net/ExpandDims:0': batch})
        pred_inc_arr[start:end] = pred.reshape(batch_size, -1)
        pred_soft_arr.append(pred_soft)
    if verbose:
        print(" done")

    return pred_inc_arr, pred_soft_arr
#-------------------------------------------------------------------------------


def _get_IS_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o._shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name(
        "FID_Inception_Net/softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)
    return pool3, softmax


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        # warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        # print("Added eps eye to covmean")

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
#-------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, batch_size=32, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    -- pool3_activations : activations at pool3 layer after feeding the images
    -- softmax_activations : activations at the output (softmax) of inception
    net after feeding the images
    """
    assert type(images[0]) == np.ndarray, "Not a numpy array"
    assert len(images[0].shape) == 3, "Images have less than 3 dimensions"
    assert np.min(images[0]) >= 0.0, "Min is less than 0"
    pool3_activations, softmax_activations = get_activations(
        images, sess, batch_size, verbose)

    mu = np.mean(pool3_activations, axis=0)
    sigma = np.cov(pool3_activations, rowvar=False)

    return mu, sigma, pool3_activations, softmax_activations


def calculate_fid_IS_kid(images, sess, mu_real=None, sigma_real=None, activations_real=None, batch_size=32, verbose=False):
    """Calculation of the FID, IS and KID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- fid   : Frechet Inception Distance
    -- IS    : Inception Score
    -- kid   : Kernel Inception Distance
    """

    fid_stats_file = "./tmp/"
    fid_stats_file += FLAGS.dataset + "_stats.npz"
    mu, sigma, pool3_activations, softmax_activations = calculate_activation_statistics(images, sess, batch_size=batch_size, verbose=verbose)
    if not (mu_real and sigma_real and activations_real):
        f = np.load(fid_stats_file)
        mu_real, sigma_real, activations_real = f['mu'][:], f['sigma'][:], f['activations_pool3'][:]
        f.close()
    splits = 10
    preds_IS = np.concatenate(softmax_activations, 0)
    scores = []
    for i in range(splits):
        part = preds_IS[(i * preds_IS.shape[0] // splits)
                         :((i + 1) * preds_IS.shape[0] // splits), :]
        kl = part * (np.log(part) -
                     np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    # Inception_score below
    IS_mean, IS_std = np.mean(scores), np.std(scores)
    # FID score below:
    mu = np.mean(pool3_activations, axis=0)
    sigma = np.cov(pool3_activations, rowvar=False)
    fid = calculate_frechet_distance(mu, sigma, mu_real, sigma_real)
    # KID score below:
    # TODO: Reset subset_size to 1000 as it was before.
    kid = polynomial_mmd_averages(pool3_activations, activations_real, n_subsets=50,
                            subset_size=1000, ret_var=False, output=sys.stdout)
    return fid, IS_mean, IS_std, kid

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
#-------------------------------------------------------------------------------


def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)


def _handle_path(path, sess):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        x = np.array([imread(str(fn)).astype(np.float32) for fn in files])
        m, s, _, _ = calculate_activation_statistics(x, sess)
    return m, s


def calculate_fid_given_paths(paths, inception_path):
    ''' Calculates the FID of two paths. '''
    inception_path = check_or_download_inception(inception_path)

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    create_inception_graph(str(inception_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = _handle_path(paths[0], sess)
        m2, s2 = _handle_path(paths[1], sess)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, nargs=2,
                        help='Path to the generated images or to .npz statistic files')
    parser.add_argument("-i", "--inception", type=str, default=None,
                        help='Path to Inception model (will be downloaded if not provided)')
    parser.add_argument("--gpu", default="", type=str,
                        help='GPU to use (leave blank for CPU only)')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    fid_value = calculate_fid_given_paths(args.path, args.inception)
    print("FID: ", fid_value)
