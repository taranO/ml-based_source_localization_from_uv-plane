import os

import numpy as np
import math
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.transform import rescale
from abc import ABC, abstractmethod
import tensorflow as tf

# =====================================================================================================================
class DatasetBase(ABC):
    def __init__(self):
        self._seed = -1

    @abstractmethod
    def _initDataSets(self):
        pass

    @abstractmethod
    def getLoader(self):
        pass

    def normaliseDynamicRange(self, image):
        '''
        The dynamic range stretching

        :param image: an image to normalize
        :return: ndarray
        '''
        image -= np.min(image, axis=(0, 1), keepdims=True)
        image /= np.max(image, axis=(0, 1), keepdims=True)

        return image

    def reshapeData(self, data, shape):
        '''
        Data reshaping

        :param data: the data_10k to reshape
        :param shape: the expected shape
        :return: ndarray
        '''
        return data.reshape((-1, (*shape)))

    def makeDir(self, dir):
        '''
        Create a directory if not exist
        :param dir: the directory path to create
        :return: str
        '''
        if not os.path.exists(dir):
            os.makedirs(dir)

        return dir

    def _trainValTestSplit(self, N, train_ratio=0.5, test_ratio=0.5):
        '''
        Split the data_10k into train, test and validation subsets

        :param N: int, total number of data_10k
        :param train_ratio: float
        :param test_ratio: float
        :return: ndarrays, train, test, validation indices
        '''

        if self._seed != -1:
            np.random.seed(seed=self._seed)

        indices = np.arange(N)
        np.random.shuffle(indices)
        N_train = math.floor(N * train_ratio)
        N_test = math.floor(N * test_ratio)

        return indices[:N_train], indices[N_train:N_train + N_test], indices[N_train + N_test:]

    def _random_color_jitter(self, sample, prob=0.3):
        flag = False
        if np.random.random_sample() < prob:
            flag = True
            sample = tf.image.random_hue(sample, 0.08)
            sample = tf.image.random_saturation(sample, 0.6, 1.6)
            sample = tf.image.random_brightness(sample, 0.05)
            sample = tf.image.random_contrast(sample, 0.7, 1.3)

        return sample, flag

    def _random_gayscale(self, sample, prob=0.25):
        flag = False
        if np.random.random_sample() < prob:
            flag = True
            sample = rgb2gray(sample)

        return sample, flag

    def _random_gaussian_blur(self, sample, min_sigma=0.1, max_sigma=2.0, prob=0.25):
        flag = False
        if np.random.random_sample() < prob:
            flag = True
            sigma = (max_sigma - min_sigma) * np.random.random_sample() + min_sigma
            sample = gaussian(sample, sigma, mode='nearest', multichannel=True)

        return sample, flag

    def _random_wave(self, sample, prob=0.25):
        flag = False
        if np.random.random_sample() < prob:
            sample[:, 1] += 10 * np.sin(2 * np.pi * sample[:, 0] / 64)

        return sample, flag

    def _scale_images(self, samples, scale_ratio=1):
        if scale_ratio <= 0:
            raise ValueError("Scaling factor should be bigger than zero.")

        if scale_ratio == 1:
            return samples

        n = samples.shape[0]
        rescaled_samples = []
        for i in range(n):
            rescaled_samples.append(rescale(samples[i], scale_ratio, multichannel=True, anti_aliasing=True))

        return np.asarray(rescaled_samples)

    def _loadNpy(self, file, allow_pickle=False):
        data = []
        with open(file, 'rb') as f:
            data = np.load(f, allow_pickle=allow_pickle)
        return data
