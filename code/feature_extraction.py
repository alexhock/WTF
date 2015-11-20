__author__ = 'ah14aeb'
import sys
import numpy
import config
import log
from helpers.radialprofile import RadialProfile
from helpers.feature_helper import FeatureHelper
from helpers import normalisation
from helpers import plots
from helpers import imagedatahelper
from helpers import boundingbox

class FeatureMatrixProcessor(object):

    def apply_options_to_general_feature_matrix(self, options, samples, mean, std):

        logger.info("applying options to general feature matrix")

        logger.info("natural log raw samples")
        logged_samples = numpy.array(samples, copy=True)
        #logged_samples = numpy.log(logged_samples)

        logger.info("normalizing logged features")
        normalized_logged_samples = numpy.array(logged_samples, copy=True)
        normalized_logged_samples, new_mean, new_std = normalisation.normalize_general_features(
            normalized_logged_samples, mean, std)
        logger.info("=============model mean and std ==============")
        logger.info(mean)
        logger.info(std)
        logger.info("=============general mean and std ==============")
        logger.info(new_mean)
        logger.info(new_std)
        logger.info("================================================")

        return logged_samples, normalized_logged_samples

    def get_samples_stats(self, options, samples):

        logger.info("get stats samples")

        logged_samples = numpy.array(samples, copy=True)
        #logged_samples = numpy.log(logged_samples)

        normalized_logged_samples, mean, std = normalisation.normalize_model_features(logged_samples)

        return mean, std


class PatchFactory(object):

    def __init__(self, image_data_helper):
        self.image_data_helper = image_data_helper

    def print_title(self):
        logger.info("Image folder path: {0}".format(self.image_data_helper._image_folder_path))

    def get_all_samples(self, options, bounding_box, all_zero):

        samples = []
        positions = []
        window = options.window_size
        radial_bin_size = options.radial_width

        image_size = [window*2, window*2]
        radial_profile = RadialProfile(image_shape=image_size, bin_size=radial_bin_size)
        feature_factory = FeatureHelper(radial_profile)

        counter = 0
        zero_counter = 0
        for x in bounding_box.x_pixel_range:
            for y in bounding_box.y_pixel_range:

                #patches, patch_positions, filters = self.image_data_helper.get_patch_all_images_with_check(
                #    x, y, window, all_zero)
                patches, patch_positions, filters = self.image_data_helper.get_patch_all_images(x, y, window)

                if patches is None:
                    counter += 1
                    zero_counter += 1
                    continue

                sample, num_ps_values = feature_factory.get_power_spectrum_all_wavebands(patches)
                if sample[0] is numpy.nan:
                    print patches[0]
                    print sample

                samples.append(sample.tolist())

                positions.append(patch_positions[0])

                if counter % 3000 == 0:
                    logger.info("processed pixels:{0} non_zero_patches: {1} x: {2} y: {3}".format(
                        counter, counter-zero_counter, x, y))
                counter += 1

        return numpy.array(samples), numpy.array(positions)


    def get_features_all_pixels(self, options, bounding_box, all_zero):
        width = bounding_box.width
        height = bounding_box.height

        logger.info("Extracting up to {0} samples from image size w: {1} h: {2}".format(width*height, width, height))
        samples, positions = self.get_all_samples(
            options=options, bounding_box=bounding_box, all_zero=all_zero)
        logger.info("samples shape: {0}  positions shape: {1}".format(samples.shape, positions.shape))

        return samples, positions


class FeatureOptions(object):

    def __init__(self, config):
        self.is_model = False
        if config['is_model'] in ['True', 'Y', 'y', 'true', '1']:
            self.is_model = True

        self.window_size = int(config['window_size'])
        self.slide = int(config['slide'])
        self.radial_width = float(config['radial_width'])
        self.output_folder_path = config['output_folder_path']
        if config['all_zero'] in ['True', 'Y', 'y', 'true', '1']:
            self.all_zero = True
        else:
            self.all_zero = False
        self.image_folder_path = config['image_folder_path']
        self.model_folder_path = config['model_folder_path']
        self.left = int(config['left'])
        self.right = int(config['right'])
        self.top = int(config['top'])
        self.bottom = int(config['bottom'])

    def print_options(self):
        logger.info("window size: {0} ".format(self.window_size))
        logger.info("radial average width: {0}".format(self.radial_width))
        logger.info("output folder path: {0}".format(self.output_folder_path))
        logger.info("model folder path: {0}".format(self.model_folder_path))
        logger.info("all_zero: {0}".format(self.all_zero))
        logger.info("image_folder_path: {0}".format(self.image_folder_path))
        logger.info("is_model: {0}".format(self.is_model))
        logger.info("left: {0} right: {1} bottom: {2} top: {3}".format(self.left, self.right, self.bottom, self.top))


def get_stats(options, matrix_processor, samples):

    mean, std = None, None

    if options.is_model:
        mean, std = matrix_processor.get_samples_stats(options, samples)
        numpy.savetxt(options.output_folder_path + "/model_mean.txt", mean, delimiter=",")
        numpy.savetxt(options.output_folder_path + "/model_std.txt", std, delimiter=",")
    else:
        # apply mean, std to the feature data
        mean = numpy.loadtxt(options.model_folder_path + "/model_mean.txt", delimiter=",")
        std = numpy.loadtxt(options.model_folder_path + "/model_std.txt", delimiter=",")

    return mean, std


def main(config):

    logger.info(__file__)

    options = FeatureOptions(config)
    options.print_options()

    matrix_processor = FeatureMatrixProcessor()

    field_bb = boundingbox.BoundingBox(
        left=options.left, right=options.right, top=options.top, bottom=options.bottom, step=options.slide)
    idh = imagedatahelper.ImageDataHelper(logger, options.image_folder_path, extension="_ok.fits")

    #sigmas = config['sigmas']
    #sigma_multipliers = config['sigma_multipliers']
    #idh.set_thresholds(sigmas, sigma_multipliers)

    field_image = PatchFactory(idh)

    # get the patches/sub images, positions and offsets (if multiple fields)
    gen_samples, gen_positions = field_image.get_features_all_pixels(options, field_bb, options.all_zero)

    logger.info("saving raw samples and positions")
    numpy.savetxt(options.output_folder_path + "/raw_gen_samples.csv", gen_samples, delimiter=",")
    numpy.savetxt(options.output_folder_path + "/gen_positions.csv", gen_positions, delimiter=",")

    # fix for natural log -- before applying natural logarithm convert 0s to very small values below the minimum
    # existing value
    #min_sample_value = numpy.min(gen_samples[gen_samples > 0])
    #gen_samples[gen_samples == 0] = (min_sample_value * 0.9)

    mean, std = get_stats(options, matrix_processor, gen_samples)

    logged_samples, normalized_logged_samples = \
        matrix_processor.apply_options_to_general_feature_matrix(options, gen_samples, mean, std)

    logger.info("Saving samples and positions...")
    numpy.savetxt(options.output_folder_path + "/normalized_samples.csv",
                  normalized_logged_samples, delimiter=",")
    #numpy.savetxt(options.output_folder_path + "/samples.csv", logged_samples, delimiter=",")

    # visualize feature matrix using pca
    plots.pca_and_plot(options.output_folder_path + '/normalized_samples_pca.png',
                 normalized_logged_samples, show_plot=False)


# #######################################################################################
# # Main
# #######################################################################################

logger = None

if __name__ == "__main__":

    config = config.get_config(sys.argv)
    log.configure_logging(config['log_file_path'])
    logger = log.get_logger("feature_extraction")

    logger.debug("*** Starting ***")
    main(config)
    logger.debug("*** Finished ***")

