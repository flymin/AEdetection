import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import logging

from misc.utils import isfloat
from misc.median import median_random_filter as median_random_filter_pt
from misc.median import median_random_pos_size_filter as median_random_pos_size_filter_pt
from misc.median import median_random_size_filter as median_random_size_filter_pt


def _reduce_precision(x, npp):
    """
    Reduce the precision of image, the numpy version.
    :param x: a float tensor, which has been scaled to [0, 1].
    :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
    :return: a tensor representing image(s) with lower precision. [0, 1]
    """
    # Note: Starting from 0, 8-bit is 0-255, npp=256 here.
    npp_int = npp - 1
    x_int = torch.round(x * npp_int)
    x_float = x_int / npp_int
    return x_float


def _random_noise(x, stddev):
    if stddev == 0.:
        rand_array = torch.zeros(x.shape)
    else:
        rand_array = torch.normal(mean=0., std=stddev, size=x.shape)
    x_random = x + rand_array
    return x_random


def bit_depth(x, bits):
    precisions = 2**bits
    return _reduce_precision(x, precisions)


def bit_depth_random(x, bits, stddev):
    x_random = _random_noise(x, stddev)
    return bit_depth(x_random, bits)


def binary_filter(x, threshold):
    x_bin = F.relu(torch.sign(x - threshold))
    return x_bin


def binary_random_filter(x, threshold, stddev=0.125):
    x_random = _random_noise(x, stddev)
    x_bin = binary_filter(x_random, threshold)
    return x_bin


def median_filter(x: torch.Tensor, width, height=-1):
    """
    Median smoothing by Scipy.
    :param x: a tensor of image(s)
    :param width: the width of the sliding window (number of pixels)
    :param height: the height of the window. The same as width by default.
    :return: a modified tensor with the same shape as x.
    """
    if height == -1:
        height = width
    x_np = x.permute(0, 2, 3, 1).cpu().numpy()  # NCHW -> NHWC
    x_np = ndimage.filters.median_filter(x_np, size=(1, width, height, 1),
                                         mode='reflect')
    x = torch.tensor(x_np).permute(0, 3, 1, 2).to(x.device)  # -> NCHW
    return x


def median_random_filter(x, width, height=-1):
    # assert False
    return median_random_filter_pt(x, width, height)


def median_random_pos_size_filter(x):
    # useless
    res = median_random_pos_size_filter_pt(x)
    return res.eval()


def median_random_size_filter(x):
    # assert False
    res = median_random_size_filter_pt(x)
    return res.eval()


# Squeezers implemented in OpenCV
# OpenCV expects uint8 as image data type.
def _opencv_wrapper(imgs: torch.Tensor, opencv_func, argv):
    ret_imgs = []
    imgs_copy = imgs.cpu().permute(0, 2, 3, 1).numpy()

    if imgs.shape[3] == 1:
        imgs_copy = np.squeeze(imgs)

    for img in imgs_copy:
        img_uint8 = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
        ret_img = opencv_func(*[img_uint8] + argv)
        if type(ret_img) == tuple:
            ret_img = ret_img[1]
        ret_img = ret_img.astype(np.float32) / 255.
        ret_imgs.append(ret_img)
    ret_imgs = np.stack(ret_imgs)

    if imgs.shape[3] == 1:
        ret_imgs = np.expand_dims(ret_imgs, axis=3)

    ret_imgs = torch.tensor(ret_imgs).permute(0, 3, 1, 2).to(imgs.device)
    return ret_imgs


# Binary filters.
def adaptive_binarize(x, block_size=5, C=33.8):
    "Works like an edge detector."
    # ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C
    # THRESH_BINARY, THRESH_BINARY_INV
    import cv2
    ret_imgs = _opencv_wrapper(x, cv2.adaptiveThreshold, [
        255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C])
    return ret_imgs


def otsu_binarize(x):
    # func = lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # return opencv_binarize(x, func)
    import cv2
    ret_imgs = _opencv_wrapper(
        x, cv2.threshold, [0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU])
    return ret_imgs


# Non-local Means
def non_local_means_bw(imgs, search_window, block_size, photo_render):
    import cv2
    ret_imgs = _opencv_wrapper(
        imgs, cv2.fastNlMeansDenoising,
        [None, photo_render, block_size, search_window])
    return ret_imgs


def non_local_means_color(imgs, search_window, block_size, photo_render):
    import cv2
    ret_imgs = _opencv_wrapper(
        imgs, cv2.fastNlMeansDenoisingColored,
        [None, photo_render, photo_render, block_size, search_window])
    return ret_imgs


def bilateral_filter(imgs, d, sigmaSpace, sigmaColor):
    """
    :param d: Diameter of each pixel neighborhood that is used during filtering. 
        If it is non-positive, it is computed from sigmaSpace.
    :param sigmaSpace: Filter sigma in the coordinate space. 
        A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). 
        When d>0, it specifies the neighborhood size regardless of sigmaSpace. 
        Otherwise, d is proportional to sigmaSpace.
    :param sigmaColor: Filter sigma in the color space. 
        A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color.
    """
    import cv2
    return _opencv_wrapper(imgs, cv2.bilateralFilter,
                           [d, sigmaColor, sigmaSpace])


# Adaptive Bilateral Filter
# https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#adaptivebilateralfilter
# Removed in OpenCV > 3.0.
def adaptive_bilateral_filter(imgs, ksize, sigmaSpace, maxSigmaColor=20.0):
    import cv2
    return _opencv_wrapper(
        imgs, cv2.adaptiveBilateralFilter,
        [(ksize, ksize),
         sigmaSpace, maxSigmaColor])


def none_func(x): return x


# Construct a name search function.
def parse_params(params_str):
    params = []

    for param in params_str.split('_'):
        param = param.strip()
        if param.isdigit():
            param = int(param)
        elif isfloat(param):
            param = float(param)
        else:
            continue
        params.append(param)

    return params


class SqueezerWrapper:
    def __init__(self, func, name, x_min=0., x_max=1.) -> None:
        self.func = func
        self.name = name
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, x):
        if x.min() < self.x_min or x.max() > self.x_max:
            logging.warning(
                "[{} Squeezer] The input data is out of data range".format(
                    self.name))
        res = self.func(x)
        # This clamp it to prevent out-of-range in sequential squeezer.
        res = torch.clamp(res, self.x_min, self.x_max)
        return res


def get_squeezer_by_name(name):
    squeezer_list = ['none_func',
                     'bit_depth',
                     'bit_depth_random',
                     'binary_filter',
                     'binary_random_filter',
                     'median_filter',
                     'median_random_filter',
                     'median_random_size_filter',
                     'adaptive_binarize',
                     'otsu_binarize',
                     'non_local_means_bw',
                     'non_local_means_color',
                     'bilateral_filter',
                     'adaptive_bilateral_filter',
                     ]
    for squeezer_name in squeezer_list:
        if name.startswith(squeezer_name):
            func_name = squeezer_name
            params_str = name[len(squeezer_name):]
            # Return a list
            args = parse_params(params_str)
            # print ("params_str: %s, args: %s" % (params_str, args))
            return SqueezerWrapper(
                lambda x: globals()[func_name](*([x] + args)),
                func_name)
    raise Exception('Unknown squeezer name: %s' % name)
