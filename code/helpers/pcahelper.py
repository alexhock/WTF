__author__ = 'AlexH'

import mdp
import scipy
#from scipy import linalg
from numpy import linalg

import numpy as np
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.utils import validation

from sklearn.decomposition import PCA

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
#import mesh_to_coords as mtc
import matplotlib.cm as cm
from matplotlib.patches import Ellipse as Ell
from matplotlib.patches import Rectangle


def run_pca_with_variance(data, n_components=2, whiten=False):
    pca = decomposition.PCA(n_components=n_components) ## specify 0 < x < 1 percentage variance required.
    # Will return components up to the variance specified
    pca.whiten = whiten
    data_reduced = pca.fit_transform(data)
    print pca.explained_variance_ratio_
    print "Variance retained 2 PCs {0}".format(((pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]) * 100))
    print "Variance retained 3 PCs {0}".format(((pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1] + pca.explained_variance_ratio_[2]) * 100))

    return pca, data_reduced

def array2d(X, dtype=None, order=None, force_all_finite=True):
    """Returns at least 2-d array with data from X"""

    X_2d = np.asarray(np.atleast_2d(X), dtype=dtype, order=order)
    if force_all_finite:
        if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
                and not np.isfinite(X).all()):
            raise ValueError("Array contains NaN or infinity.")
    return X_2d

def evd(X):

    data = np.copy(X)
    # subtract the mean
    data -= np.mean(data, axis=0)

    sigma = np.cov(X)
    #sigma = np.dot(data, data.T)
    #sigma = np.divide(sigma, (data.shape[1]-1))

    eigenvals, eigenvecs = linalg.eig(sigma)
    return eigenvals, eigenvecs


def evd_transform(X, components, ncomps=-1):
    X = validation.array2d(X)
    if ncomps == -1:
        X_transformed = np.dot(X, components)
        #X_transformed = np.dot(components.T, X.T)
    else:
        X_transformed = np.dot(X, components[0:ncomps])
    return X_transformed


def native_pca(X):
    n_samples, n_features = X.shape

    Y = X / np.sqrt(X.shape[0]-1)

    U, S, V = linalg.svd(Y, full_matrices=False)

    # print variances
    get_native_variance(S, n_samples)

    return (U, S, V)

def get_native_variance(S, n_samples):
    explained_variance_ = (S ** 2) / n_samples
    explained_variance_ratio_ = (explained_variance_ /
                                  explained_variance_.sum())
    #print explained_variance_ratio_[0:10]
    #print explained_variance_ratio_[0:2].sum()

    pc_ratios = []
    for ratio in explained_variance_ratio_:
        pc_ratios.append(int(ratio * 10000) / 100.0)

    print "variances: {0}".format(pc_ratios)
    #print "Variance retained 1 PCs {0}".format(((pc_ratios[0]) * 100))
    #print "Variance retained 2 PCs {0}".format(((pc_ratios[0] + pc_ratios[1]) * 100))
    #print "Variance retained 3 PCs {0}".format(((pc_ratios[0] + pc_ratios[1] + pc_ratios[2]) * 100))
    return pc_ratios #explained_variance_ratio_

def native_transform(X, components, ncomps=-1):
    #X = validation.array2d(X)
    X = np.array(X, copy=True)
    if ncomps == -1:
        X_transformed = np.dot(X, components.T)
        #X_transformed = np.dot(components.T, X.T)
    else:
        X_transformed = np.dot(X, components[0:ncomps].T)

    return X_transformed

def native_inverse_transform(X, components):
    X_transformed = np.dot(X, components)
    return X_transformed


def native_fit_transform(X):
    U, S, V = native_pca(X)
    #U *= S[:self.n_components]
    U *= S

    return U

def normalize_data(data):
    sigma = data.std()
    mu = np.mean(data)
    normed_data = data - mu
    normed_data = np.divide(normed_data, sigma)  # element wise division
    return mu, sigma, normed_data



def mean_normalization(data):
    mu = np.mean(data)
    return mu, (data - mu)


def normalize_image(data, mu, sigma):
    normed_data = data - mu
    normed_data = np.divide(normed_data, sigma)  # element wise division
    return normed_data

def mean_normalize_features(data):
    temp_data = np.array(data, copy=True)
    mean = np.mean(temp_data, axis=0)
    temp_data -= mean
    return temp_data

def tdif_norm(data):

    totals = np.sum(data, axis=1)
    data = np.divide(data.T, totals)
    #data = np.divide(data, totals)
    return data.T

def normalize_features(patches):

    temp_data = np.array(patches)

    mean1_ = np.mean(temp_data, axis=0)
    std1_ = np.std(temp_data, axis=0)

    # normalize
    temp_data -= mean1_
    temp_data /= std1_

    mean2_ = np.mean(temp_data, axis=0)
    std2_ = np.std(temp_data, axis=0)

    ncols = temp_data.shape[1]
    for i in range(ncols):
        print "{0}  & {1}     & {2}     & {3}   & {4}".format(i+1, mean1_[i], std1_[i], mean2_[i], std2_[i])

    return temp_data

def normalize_features2(patches):

    temp_data = np.array(patches)

    ncols = temp_data.shape[1]

    printables = np.zeros([ncols, ncols*2])

    for i in range(ncols):
        mu = np.mean(temp_data[:, i])
        sigma = np.std(temp_data[:, i])
        printables[i, 0] = mu
        printables[i, 1] = sigma
        #print "{0}  & {1}     & {2}".format(i+1, mu, sigma)
        #print "mean col: {0} mean val: {1}".format(i, numpy.mean(temp_data[:,i]))
        #print "std col: {0} std val: {1}".format(i, numpy.std(temp_data[:,i]))

    # normalize
    mean1_ = np.mean(temp_data, axis=0)
    std1_ = np.std(temp_data, axis=0)

    temp_data -= mean1_
    temp_data /= std1_

    ## calculate
    for i in range(ncols):
        mu = np.mean(temp_data[:, i])
        sigma = np.std(temp_data[:, i])
        printables[i, 2] = mu
        printables[i, 3] = sigma

    for i in range(ncols):
        print "{0}  & {1}     & {2}     & {3}   & {4}".format(i+1, printables[i, 0], printables[i, 1],
                                                              printables[i, 2], printables[i, 3])
    return temp_data



def remove_outliers_norm_denorm(samples, images, outlier_threshold):

    print "removing outliers. Threshold: {0} shape: {1}".format(outlier_threshold, samples.shape)
    # normalize
    mean1_ = np.mean(samples, axis=0)
    std1_ = np.std(samples, axis=0)
    samples -= mean1_
    #samples /= std1_

    print "after norm samples shape: {0}".format(samples.shape)

    num_outliers_one = samples[samples[:, 0] > outlier_threshold].shape[0]
    num_outliers_one += samples[samples[:, 0] < (outlier_threshold*-1)].shape[0]

    print "num outliers: {0}".format(num_outliers_one)

    print std1_
    #samples = remove_outliers(samples, outlier_threshold)
    num_features = samples.shape[1]
    for i in range(num_features):
        limit = (std1_[i]*outlier_threshold)
        neg_limit = (limit * (-1))
        min = np.min(samples[:, i])
        max = np.max(samples[:, i])

        samples_greater_than_limit = samples[samples[:, i] > limit].shape[0]
        samples_less_than_limit = samples[samples[:, i] < (limit*-1)].shape[0]
        print "feature {0} std: {1}  limit:{2}  neg_limit: {3} min:{4} max:{5}  greater: {6}  less than: {7}".format(i, std1_[i], limit, neg_limit, min, max, samples_greater_than_limit, samples_less_than_limit)
        bef = samples.shape[0]
        befi = images.shape[0]

        less_than_threshold_mask = samples[:, i] < limit
        samples = samples[less_than_threshold_mask]
        images = images[less_than_threshold_mask]

        #samples = samples[samples[:, i] < limit] # keep samples less than threshold
        bef1 = samples.shape[0]

        greater_than_minus_threshold_mask = samples[:, i] > neg_limit
        samples = samples[greater_than_minus_threshold_mask]
        images = images[greater_than_minus_threshold_mask]

        #samples = samples[samples[:, i] > neg_limit] # keep samples greater than -threshold
        aft1 = samples.shape[0]
        afti = images.shape[0]
        print "before: {0} after high: {1} after low: {2}  beforeimage: {3} afterimage: {4}".format(bef, bef1, aft1, befi, afti)
    print "after outlier removal samples shape: {0}".format(samples.shape)
    # reverse normalization for the remaining samples
    #samples *= std1_
    samples += mean1_

    print "after denorm samples shape: {0}".format(samples.shape)
    print "finished removing outliers"
    return samples, images


def remove_outliers(samples, outlier_threshold):

    num_samples, num_features = samples.shape
    print "number of samples before outlier removal: {0}".format(num_samples)
    for i in range(num_features):
        samples = samples[samples[:, i] < outlier_threshold] # keep samples less than threshold
        samples = samples[samples[:, i] > (outlier_threshold*-1)] # keep samples greater than -threshold
    print "number of samples after outlier removal: {0}".format(samples.shape[0])

    return samples


def squarify(image):
    width, height = image.shape
    if width == height:
        return image
    if width > height:
        new_shape = width
    else:
        new_shape = height

    square_image = np.zeros([new_shape, new_shape])
    square_image[0:width, 0:height] = image

    return square_image

def expand(image, width, height):

    old_width, old_height = image.shape
    if old_width >= width or old_height >= height:
        print "image already bigger than requested"
        return image

    offsetx = (width - old_width) / 2
    offsety = (height - old_height) / 2

    expanded_image = np.zeros([height, width])
    expanded_image[offsetx:offsetx+old_width, offsety:offsety+old_height] = image

    return expanded_image

def squarify_on_xy(image, centerx, centery):

    width, height = image.shape
    if width == height:
        return image

    newcenterx = centerx
    newcentery = centery
    offsetx = 0
    offsety = 0

    if width > height:
        # expand height
        offsety = (width - height) / 2
        newcentery += offsety
        new_shape = width
    else:
        # expand width
        offsetx = (height - width) / 2
        newcenterx += offsetx
        new_shape = height

    square_image = np.zeros([new_shape, new_shape])
    square_image[offsetx:offsetx+width, offsety:offsety+height] = image

    return square_image#, newcenterx, newcentery



def whiten(X,fudge=1E-18):

   # the matrix X should be observations-by-components

   # get the covariance matrix
   Xcov = np.dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d, V = np.linalg.eigh(Xcov)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = np.diag(1. / np.sqrt(d+fudge))

   # whitening matrix
   W = np.dot(np.dot(V, D), V.T)

   # multiply by the whitening matrix
   X_white = np.dot(X, W)

   return X_white, W

def svd_whiten(X):

    U, s, Vt = np.linalg.svd(X)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white