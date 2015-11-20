__author__ = 'ah14aeb'
import os
import numpy as np
import random
import re
from astropy.io import fits as pyfits

class ImageDataHelper(object):

    _num_images = 0
    _image_paths = None
    _image_files = None
    _image_range = None
    _image_shape = None
    _image_means = []
    _gal_clusters = []
    _wavelengths = []
    _sigmas = []
    _sigma_multipliers = []
    _thresholds = []


    def __init__(self, logger, image_folder_path, extension=".fits", use_mem_map=False):

        self.logger = logger
        self._image_folder_path = image_folder_path

        training_image_paths = []
        image_files = os.listdir(image_folder_path)
        for i, t in enumerate(image_files):
            if t.endswith(extension) is False:
                logger.info("Skipping file: {0} as it does not match extension: {1}".format(t, extension))
                continue
            training_image_paths.append(t)

        training_image_paths.sort(reverse=True)

        image_data = []
        for image_iter in range(len(training_image_paths)):
            training_image_path = image_folder_path + training_image_paths[image_iter]

            pic = self.__get_fits_image_matrix(training_image_path)

            if use_mem_map is False:
                # load the images into a numpy array in RAM
                # calc mean forces load of the whole image into ram
                mu = np.mean(pic)

            image_data.append(pic)

            if self._image_shape is None:
                self._image_shape = pic.shape
            if pic.shape != self._image_shape:
                logger.warning("Error fits are different sizes: {0} {1} {2}".format(
                    self._image_shape, pic.shape, training_image_path))

            logger.info("Index: {0} Image Name: {1}".format(image_iter, training_image_paths[image_iter]))

        for i in range(len(training_image_paths)):
            #CDFSmosaic_allch_atlas_radio_
            m = re.match("CDFSmosaic_allch_([a-z]+)_([a-z]+)_", training_image_paths[i])
            self._gal_clusters.append(m.groups(0)[0])
            self._wavelengths.append((m.groups(0)[1]))

        self._image_paths = np.array(training_image_paths)
        self._image_files = image_data  # don't add to numpy as it will load into RAM leave as memory mapped files.
        self._num_images = len(self._image_paths)
        self._image_range = range(self._num_images)

    @staticmethod
    def __get_fits_image_matrix(filename=''):
        img = pyfits.open(filename)
        img_data = img[0].data[0][0]  # get the image
        return img_data  ## return the memory mapped file

    def get_wavelengths(self):
        return np.array(self._wavelengths)

    def get_sigmas(self):
        return np.array(self._sigmas)

    def get_thresholds(self):
        return np.array(self._thresholds)

    def set_thresholds(self, sigmas, sigma_multipliers):
        self.logger.info("setting sigmas: {0}, wavelength order: {1}".format(sigmas, self._wavelengths))
        new_sigmas = []
        new_sigma_multipliers = []
        new_thresholds = []
        # load them in the order of the wavelengths to ensure the right sigma matches the right wavelength
        for wavelength in self._wavelengths:
            sigma = sigmas[wavelength]
            sigma_multiplier = sigma_multipliers[wavelength]
            threshold = float(sigma) * float(sigma_multiplier)
            new_sigmas.append(sigma)
            new_sigma_multipliers.append(sigma_multiplier)
            new_thresholds.append(threshold)

        self.logger.info("new sigmas: {0} wavelength order: {1}". format(new_sigmas, self._wavelengths))
        self.logger.info("new sigma multipliers: {0} wavelength order: {1}". format(
            new_sigma_multipliers, self._wavelengths))
        self.logger.info("new thresholds: {0} wavelength order: {1}". format(new_thresholds, self._wavelengths))
        self._sigmas = new_sigmas
        self._sigma_multipliers = new_sigma_multipliers
        self._thresholds = new_thresholds

    def get_image(self, image_idx):
        return self._image_files[image_idx]

    def get_image_count(self):
        return self._num_images

    def get_image_shape(self):
        return self._image_shape

    def get_image_mean(self, image_index):
        return self._image_means[image_index]

    def get_patch(self, position, window_size):
        image_index = position[0]
        x = position[1]
        y = position[2]
        #print "x: {0} y:{1}".format(x, y)
        return self.__get_clip(image_index, x, y, window_size)

    def get_pixel(self, position):
        image_index = position[0]
        x = position[1]
        y = position[2]
        image = self._image_files[image_index]
        return image[y, x]

    def get_rectangle(self, image_index, left, right, top, bottom):
        """
        :type image_index: int
        :param left:
        :param right:
        :param top:
        :param bottom:
        :return:
        """
        image = self._image_files[image_index]
        return image[bottom:top, left:right]

    def __is_zero(self, x, y):
        is_zero = False
        for i in range(self._num_images):
            if self._image_files[i][y, x] == 0:
                is_zero = True
                break
        return is_zero

    def __are_all_zero(self, x, y):
        is_zero = True
        for i in range(self._num_images):
            if self._image_files[i][y, x] > 0:
                is_zero = False
                break
        return is_zero

    def __get_clip(self, image_index, x, y, window_size):

        image = self._image_files[image_index]

        top = y + window_size
        bottom = y - window_size
        left = x - window_size
        right = x + window_size

        return image[bottom:top, left:right]

    @staticmethod
    def set_patch_rect(image, position, window_size, color):
        image_index = position[0]
        x = position[1]
        y = position[2]

        top = y + window_size
        bottom = y - window_size
        left = x - window_size
        right = x + window_size

        image[bottom:top, left:right] = color

    def get_patch_all_images_from_pos(self, position, window_size):
        return self.get_patch_all_images(position[1], position[2], window_size)

    def get_patch_all_images_with_check(self, xpos, ypos, window_size, all_zero=True):

        if all_zero == True and self.__is_zero(xpos, ypos) == True:
            return None, None, None

        if all_zero == False and self.__are_all_zero(xpos, ypos) == True:
            return None, None, None

        return self.get_patch_all_images(xpos, ypos, window_size)

    def get_patch_all_images(self, xpos, ypos, window_size):
        patches = []
        positions = []
        filters = []
        for i in range(self._num_images):
            position = np.array([0, 0, 0])
            position[0] = i
            position[1] = xpos
            position[2] = ypos
            r_clip = self.get_patch(position, window_size)
            patches.append(r_clip)
            positions.append(position)
            filters.append(self._wavelengths[i])
        return patches, positions, filters

    def get_random_patch_all_images(self, bounding_box, window_size, non_zero=False):
        # get random position
        xpos, ypos = bounding_box.get_random_position()

        count = 0
        if non_zero == True:
            while (self.__is_zero(xpos, ypos) == True):   #__are_all_zero
            #while (self.__are_all_zero(xpos, ypos) == True):
                xpos, ypos = bounding_box.get_random_position()
                count +=1

        if count > 2000:
            self.logger.info("random point non zero count: {0}".format(count))

        patches = []
        positions = []
        filters = []

        for i in range(self._num_images):
            position = np.array([0, 0, 0])
            position[0] = i # image index
            position[1] = xpos
            position[2] = ypos

            r_clip = self.get_patch(position, window_size)
            if r_clip is None:
                self.logger.error("ERROR*********")

            patches.append(r_clip)
            positions.append(position)
            filters.append(self._wavelengths[i])

        return patches, positions, filters

    def get_random_patch(self, bounding_box, window_size, zero_pixel_limit=-1):

        use_threshold = False
        if zero_pixel_limit > -1:
            use_threshold = True

        position = np.array([0, 0, 0])

        r_clip = None
        i = 0
        while i < 1000:

            i += 1

            position[0] = random.choice(self._image_range)  # random image
            position[1] = random.choice(bounding_box.xpixel_range)  # random x pos in bounding box
            position[2] = random.choice(bounding_box.ypixel_range)  # random y pos in bounding box

            r_clip = self.get_patch(position, window_size)

            if use_threshold:
                zero_pixes = r_clip[r_clip <= 0].shape[0]
                if zero_pixes > zero_pixel_limit:
                    #  print "zero_pixes: {0} limit: {1}".format(zero_pixes, _zero_pixel_limit)
                    continue

            break

        if r_clip is None:
            self.logger.error("ERROR*********")

        return r_clip, position
