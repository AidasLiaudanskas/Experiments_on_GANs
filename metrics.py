"""
Metrics module. Some functions use tensors, some numpy arrays
"""

import tensorflow as tf
import numpy as np
from scipy import linalg


def chi_square(fake_data, real_data):
    # Extra hook for debug: log chi-square distance between G's output histogram and the dataset's histogram
    value_range = [0.0, 1.0]
    nbins = 100
    # Original code:
    # hist_g = tf.histogram_fixed_width(
    #     fake_data, value_range, nbins=nbins, dtype=tf.float32) / nbins
    # hist_images = tf.histogram_fixed_width(
    #     real_data, value_range, nbins=nbins, dtype=tf.float32) / nbins
    hist_g = tf.histogram_fixed_width(
        fake_data, value_range, nbins=nbins) / nbins
    hist_images = tf.histogram_fixed_width(
        real_data, value_range, nbins=nbins) / nbins
    return tf.reduce_mean(
        tf.div(tf.square(hist_g - hist_images), hist_g + hist_images + 1e-5))


def L_norm(pre_real, pre_fake, order):
    # Returns the batch average of selected n_norm
    # 1) Take difference of latent vectors
    # 2) Take norm of that difference
    # 3) Take average of the batch of norms
    return tf.reduce_mean(tf.norm(tf.subtract(pre_real, pre_fake), ord=order, axis=1))


def frechet_distance(pre_real, pre_fake, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    Inputs: activation vectors for real data and fake data
    Returns:
    --   : The Frechet Distance.
    TODO: try to do it all in tensorflow commands
    """

    mu1 = np.mean(pre_real, axis=0)
    sigma1 = np.cov(pre_real, rowvar=False)
    mu2 = np.mean(pre_fake, axis=0)
    sigma2 = np.cov(pre_fake, rowvar=False)

    # TODO: Finish this bit
    diff = mu1 - mu2

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


def cos_distance(pre_real, pre_fake):
    # Normalize both vectors and take their dot product, then average batch
    pre_real = tf.nn.l2_normalize(pre_real, 1)
    pre_fake = tf.nn.l2_normalize(pre_fake, 1)
    # element-wise multiply (dot products) and take avg as operations are commutative
    # output should be between 0 and 1
    return tf.reduce_mean(tf.multiply(pre_real, pre_fake))
