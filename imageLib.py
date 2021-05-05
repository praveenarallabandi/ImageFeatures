# imports
import os
import toml
import numpy as np  
import numba as nb
from typing import List
import matplotlib.pyplot as plt
import time
import os
from colorama import Fore, Style
from PIL import Image # Used only for importing and exporting images

def getSingleChannel(image: np.array, colorSpectrum: str) -> np.array:
    """Get the image based on R, G or B specturm

    Args:
        image ([type]): original image
        colorSpectrum ([type]): color specturm

    Returns:
        [type]: image for single color spectrum
    """
    if(colorSpectrum == 'R') :
        img = image[:, :, 0]
        
    if(colorSpectrum == 'G') :
        img = image[:, :, 1]

    if(colorSpectrum == 'B') :
        img = image[:, :, 2]

    return img

def histogram(image: np.array) -> np.array:
    """Calculate histogram for specified image

    Args:
        image (np.array): input image
        bins ([type]): number of bins

    Returns:
        np.array: calculated histogram value
    """
    """ maxval = 255.0
    bins = np.linspace(0.0, maxval, 257)
    flatImage = image.flatten()
    vals = np.mean(flatImage, axis=0)
    # bins are defaulted to image.max and image.min values
    hist, bins2 = np.histogram(vals, bins, density=True)
    return hist, bins2 """
    hist: np.array = np.zeros(256)

    imageSize: int = len(image)

    for pixel_value in range(256):
        for i in range(imageSize):

            if image.flat[i] == pixel_value:
                hist[pixel_value] += 1

    return hist

def findMiddleHist(hist: np.array, min_count: int = 5) -> int:

    bins = len(hist)
    # print(bins)
    histogramStart = 0
    while hist[histogramStart] < min_count:
        print('hist[histogramStart] : %s histogramStart : %s', hist[histogramStart], histogramStart)
        histogramStart += 1

    histogramEnd = bins - 1
    while hist[histogramEnd] < min_count:
        histogramEnd -= 1

    maxVal = 255.0
    print('Maxval: %s', maxVal)

    histogramCenter = int(round(np.average(np.linspace(0, maxVal, bins), weights=hist)))
    left = np.sum(hist[histogramStart:histogramCenter])
    right = np.sum(hist[histogramCenter : histogramEnd + 1])

    while histogramStart < histogramEnd:
        if left > right:
            left -= hist[histogramStart]
            histogramStart += 1
        else:
            right -= hist[histogramEnd]
            histogramEnd -= 1
        calculatedCenter = int(round((histogramEnd + histogramStart) / 2))

        if calculatedCenter < histogramCenter:
            left -= hist[histogramCenter]
            right += hist[histogramCenter]
        elif calculatedCenter > histogramCenter:
            left += hist[histogramCenter]
            right -= hist[histogramCenter]

        histogramCenter = calculatedCenter

    return histogramCenter

def histogramThresholding(image: np.array, hist: np.array) -> np.array:

    if hist.any() == None:
        hist = histogram(image)
    middle = findMiddleHist(hist)
    print('middle %s', middle)
    imageCopy = image.copy()
    imageCopy[imageCopy > middle] = 255
    imageCopy[imageCopy < middle] = 0

    imageCopy = imageCopy.astype(np.uint8)

    return imageCopy.reshape(image.shape)

def erode(img_arr: np.array, window: int = 1) -> np.array:

    r = np.zeros_like(img_arr)
    [yy, xx] = np.where(img_arr > 0)

    off = np.tile(range(-window, window + 1), (2 * window + 1, 1))
    x_off = off.flatten()
    y_off = off.T.flatten()

    n = len(xx.flatten())
    x_off = np.tile(x_off, (n, 1)).flatten()
    y_off = np.tile(y_off, (n, 1)).flatten()

    ind = np.sqrt(x_off ** 2 + y_off ** 2) > window
    x_off[ind] = 0
    y_off[ind] = 0

    xx = np.tile(xx, ((2 * window + 1) ** 2))
    yy = np.tile(yy, ((2 * window + 1) ** 2))

    nx = xx + x_off
    ny = yy + y_off

    ny[ny < 0] = 0
    ny[ny > img_arr.shape[0] - 1] = img_arr.shape[0] - 1
    nx[nx < 0] = 0
    nx[nx > img_arr.shape[1] - 1] = img_arr.shape[1] - 1

    r[ny, nx] = 255

    return r.astype(np.uint8)

def dilate(img_arr: np.array, window: int = 1) -> np.array:
    inverted_img = np.invert(img_arr)
    eroded_inverse = erode(inverted_img, window).astype(np.uint8)
    eroded_img = np.invert(eroded_inverse)

    return eroded_img

def opening(image: np.array, histogram: np.array) -> np.array:
    imageThresholding = histogramThresholding(image, histogram)
    print('SegmentedImage - HistThresholding')
    eroded = erode(imageThresholding, 1)
    print('Erode - HistThresholding')
    opened = dilate(eroded, 1)
    print('Dilate - HistThresholding')

    return opened