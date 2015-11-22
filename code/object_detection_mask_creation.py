__author__ = 'ah14aeb'
import sys
import numpy as np
import config
import log
from helpers.imagedatahelper import ImageDataHelper

def main(config):

    logger.info(__file__)

    output_folder_path = config['output_folder_path']
    image_folder_path = config['image_folder_path']
    sigmas = config['sigmas']
    sigma_multipliers = config['sigma_multipliers']
    positions_path = config['positions_path']

    logger.info("positions path " + positions_path)

    # load gen_positions
    gen_positions = np.loadtxt(positions_path, delimiter=",", dtype=np.int)

    idh = ImageDataHelper(logger, image_folder_path, extension="_ok.fits")

    xv = gen_positions[:, 1]  # x
    yv = gen_positions[:, 2]  # y

    for sigma_multiplier in sigma_multipliers:
        sigma_multiplier = float(sigma_multiplier)

        masks = []
        for i in range(idh._num_images):
            sigma = config['sigmas'][idh._wavelengths[i]]
            threshold = float(sigma) * sigma_multiplier

            image_data = idh.get_image(i)
            values = image_data[yv, xv]

            mask = (values > threshold) * 1
            masks.append(mask)

            logger.info("image: {0} wavelength: {1} sigma: {2} sigma_multiplier: {3} treshold: {4}".format(
                idh._image_paths[i], idh._wavelengths[i], sigma, sigma_multiplier, threshold))

            if idh._wavelengths[i] not in idh._image_paths[i]:
                print("Huge error, wave length not in image file name: {0}".format(idh._image_paths[i]))


        masks = np.array(masks)

        # logical or to combine into one. if a pixel is greater than the threshold on any of the wavebands files then it
        # it will be allowed
        final_mask = masks[0] #np.logical_or(masks[0], masks[1])
        #final_mask = np.logical_or(final_mask, masks[2])
        if len(masks) > 1:
            print "error {0}".format(len(masks))

        np.savetxt(output_folder_path + "/sigma{0}_positions_mask.txt".format(
            int(sigma_multiplier)), final_mask, delimiter=",", fmt="%i")



# #######################################################################################
# # Main
# #######################################################################################

logger = None

if __name__ == "__main__":

    config = config.get_config(sys.argv)
    log.configure_logging(config['log_file_path'])
    logger = log.get_logger("object_detection_masks")
    logger.debug("*** Starting ***")
    main(config)
    logger.debug("*** Finished ***")


