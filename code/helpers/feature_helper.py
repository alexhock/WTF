__author__ = 'ah14aeb'

import numpy.fft as npfft
import numpy as np
import radialprofile
from radialprofile import RadialProfile

class FeatureHelper(object):

    def __init__(self, radial_profile):
        self.radial_profile = radial_profile

    def get_patch_power_spectrum(self, patch):

        FA = npfft.fft2(patch)
        FA_FBconj = npfft.fftshift(FA * np.conjugate(FA))
        trans = np.real(FA_FBconj)

        profile2 = self.radial_profile.azimuthal_average_new(trans)

        return profile2

    def get_power_spectrum_all_wavebands(self, patches):
        d_clip = np.array([])
        for i in range(len(patches)):
            patch = patches[i]
            ps = self.get_patch_power_spectrum(patch)
            num_vals = len(ps)
            ps = ps[0:-1]
            d_clip = np.concatenate((d_clip, ps), axis=0)
        return d_clip, num_vals - 1
