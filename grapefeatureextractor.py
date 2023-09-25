import torch
import cv2
import numpy as np
from PIL.Image import Image
from hog import hog
from PIL import Image
import warnings


class GrapeFeatureExtractor:
    """
    Extracts histogram features from cropped grape images.
    """

    def extract(self, images, nbin_rows, nbin_cols, cross=None, hsv=False, hog_type=None, only_hog=False, **hog_kwargs):
        """
        :param images: a single image or a list of images. These should be of type numpy.ndarray, or PIL.Image
        :param nbin_cols: number of bins in the width direction
        :param nbin_rows: number of bins in the height direction
        :param cross: None or tuple. The tuple should have four numbers: (start_row, end_row, start_col, end_col),
        where the row and col values here are referred to the histogram grid, not to the image pixels. Example: ina
        8x8 histogram grid a possible cross is [1,5,1,8] to keep the bins that are in the second to seventh column,
        or in the second to fourth row, included.
        :param hsv: boolean value telling if the image should be converted to hsv before computing the histogram.
        :param hog_type: can be None or a string. The possible values are 'image' or 'channels'. in the first case,
        the HOG features are computed on the grayscale converted image, as common practice, in the second the HOG
        features are computed on each one of the three color channels (in whatever format used: RGB, HSV...).
        :param only_hog: boolean. If True, the color histogram is not computed and only HOG features are computed
        (on the whole image, or on the color channels). If True, parameter hog cannot be None.
        :param hog_kwargs: these are the key-value parameters that can be passed to the hog function. The hog
        function has been extracted from the skimage package and modified in order to turn off the block
        normalization, since it could mess with the color gradient accuracy. Trough this kwargs it is possible to set
        some hyperparameters to effectively test the best hog vector for the current problem.
        :return:
        """
        # check input values
        if nbin_rows < 1 or nbin_cols < 1:
            raise ValueError(f'nbin_rows and nbin_cols must be grater than 0. Found {nbin_rows}, {nbin_cols}')
        if not isinstance(images, list):
            images = [images]
        for i, im in enumerate(images):
            if not (isinstance(im, Image.Image) or isinstance(im, np.ndarray)):
                raise ValueError(f"Input images should be numpy ndarray or Images. Found {type(im)}")
            # convert all formats to numpy array
            if isinstance(im, Image.Image):
                im = np.asarray(im)
            if im.dtype == np.uint8:
                im = im.astype(np.single) / 255
            images[i] = im
        # Check for correct image dimension and size
        for im in images:
            if not len(im.shape) == 3:
                raise ValueError(f"Images should be RGB, not other formats. Found len(im.shape) == {len(im.shape)}")
            # if im.shape[0] % nbin_rows or im.shape[1] % nbin_cols:
            #     raise ValueError(f"Images size should be perfectly divisible by the number of bins requested.")
        if cross:
            if not (isinstance(cross, tuple) or isinstance(cross, list)):
                raise ValueError(f"Parameter 'cross' should be an instance of tuple or list types, found {type(cross)}")
            if len(cross) != 4:
                raise ValueError(f"Parameter cross should have 4 values, found {len(cross)}")
            if cross[0] < 0 or cross[1] > nbin_rows or cross[2] < 0 or cross[3] > nbin_cols:
                raise ValueError(
                    f"Parameter cross values should be in the 0-{nbin_rows} and 0-{nbin_cols} ranges, found {cross}")
        if hog_type:
            if not (('image' == hog_type) or ('channels' == hog_type)):
                raise ValueError(f'Parameter cross should be None, "image" or "channels". Found {hog_type}')
        if only_hog and hog_type is None:
            raise ValueError("only_hog can be True only if hog_type is not None")
        if only_hog and hsv:
            warnings.warn("If only_hog is True, hsv should be False")
            #raise ValueError("If only_hog is True, hsv should be False")
        hist_batch = list()

        # extract color intensity features
        for image in images:
            # convert image to HSV color space
            if hsv:
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
                image[:, :, 0] = image[:, :, 0] / 360
            bin_h = image.shape[0] // nbin_rows
            bin_w = image.shape[1] // nbin_cols
            if hog_type:
                if 'pixels_per_cell' not in hog_kwargs.keys():
                    hog_kwargs['pixels_per_cell'] = (bin_h, bin_w)
                elif hog_kwargs['pixels_per_cell'] is None:
                    hog_kwargs['pixels_per_cell'] = (bin_h, bin_w)
                # elif hog_kwargs['pixels_per_cell'][0] > bin_h or hog_kwargs['pixels_per_cell'][1] > bin_w:
                #     hog_kwargs['pixels_per_cell'] = (bin_h, bin_w)
                #     warnings.warn(f"Wrong value for pixels_per_cell parameter. Found {hog_kwargs['pixels_per_cell']} "
                #                   f"but bin size is maximum {(bin_h, bin_w)}")

            # compute color features, if requested
            bin_hist = list()
            if not only_hog:
                for i in range(nbin_rows):
                    for j in range(nbin_cols):
                        if cross:
                            if (i < cross[0] or i >= cross[1]) and (j < cross[2] or j >= cross[3]):
                                continue
                        # compute bin
                        bin = image[i * bin_h:(i + 1) * bin_h, j * bin_w:(j + 1) * bin_w]
                        # extract color
                        avg_clr = np.mean(bin, axis=(0, 1))
                        bin_hist.append(avg_clr)

            # compute hof features, if requested
            if hog_type:
                # remove background before computing hog features
                if cross:
                    for i in range(nbin_rows):
                        for j in range(nbin_cols):
                            bin = image[i * bin_h:(i + 1) * bin_h, j * bin_w:(j + 1) * bin_w]
                            np.copyto(bin, np.zeros_like(bin))
                if hog_type == 'image':
                    # extract hog on grayscale image
                    image = GrapeFeatureExtractor.rgb2gray(image)
                    bin_hist.append(hog(image, block_norm=None, **hog_kwargs))
                elif hog_type == 'channels':
                    # extract hog for each channel of the bin
                    for c in range(3):
                        bin_hist.append(hog(image[:, :, c], block_norm=None, **hog_kwargs))

            # concatenate color and hog features for this image
            hist = np.concatenate(bin_hist)
            hist_batch.append(np.array(hist).flatten())
        # stack histograms for all samples
        hist_batch = np.array(hist_batch)
        return hist_batch

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])