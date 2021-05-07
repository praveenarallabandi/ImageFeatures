# imports
from ImageAnalysisPart1 import convertToSingleColorSpectrum
import os
import toml
import math
import numpy as np  
import numba as nb
from operator import eq
from typing import List
import matplotlib.pyplot as plt
import time
import os
from colorama import Fore, Style
from PIL import Image # Used only for importing and exporting images
from random import randrange
from math import sqrt

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

def erode(image: np.array, erosion_level: int = 3) -> np.array:

    """ structuring_kernel = np.full(shape=(erosion_level, erosion_level), fill_value=255)
    # image_src = binarize_this(image_file=image_file)

    orig_shape = image_src.shape
    pad_width = erosion_level - 2
    print('padwidth %d', pad_width)
    print('image_src %d', image_src.shape)
    # pad the matrix with `pad_width`
    image_pad = np.pad(image_src, (pad_width, pad_width), mode='constant')
    pimg_shape = image_pad.shape
    h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])

    # sub matrices of kernel size
    flat_submatrices = np.array([
        image_pad[i:(i + erosion_level), j:(j + erosion_level)]
        for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)
    ])

    # condition to replace the values - if the kernel equal to submatrix then 255 else 0
    image_erode = np.array([255 if (i == structuring_kernel).all() else 0 for i in flat_submatrices])
    image_erode = image_erode.reshape(orig_shape)

    return image_erode """
    r = np.zeros_like(image)
    [yy, xx] = np.where(image > 0)

    # prepare neighborhoods
    off = np.tile(range(-erosion_level, erosion_level + 1), (2 * erosion_level + 1, 1))
    x_off = off.flatten()
    y_off = off.T.flatten()

    # duplicate each neighborhood element for each index
    n = len(xx.flatten())
    x_off = np.tile(x_off, (n, 1)).flatten()
    y_off = np.tile(y_off, (n, 1)).flatten()

    # round out offset
    ind = np.sqrt(x_off ** 2 + y_off ** 2) > erosion_level
    x_off[ind] = 0
    y_off[ind] = 0

    # duplicate each index for each neighborhood element
    xx = np.tile(xx, ((2 * erosion_level + 1) ** 2))
    yy = np.tile(yy, ((2 * erosion_level + 1) ** 2))

    nx = xx + x_off
    ny = yy + y_off

    # bounds checking
    ny[ny < 0] = 0
    ny[ny > image.shape[0] - 1] = image.shape[0] - 1
    nx[nx < 0] = 0
    nx[nx > image.shape[1] - 1] = image.shape[1] - 1

    r[ny, nx] = 255

    return r.astype(np.uint8)

def dilate(image: np.array, window: int = 1) -> np.array:
    inverted_img = np.invert(image)
    eroded_inverse = erode(inverted_img, window).astype(np.uint8)
    eroded_img = np.invert(eroded_inverse)

    return eroded_img

def opening(image: np.array, histogram: np.array) -> np.array:
    imageThresholding = histogramThresholding(image, histogram)
    print('SegmentedImage - HistThresholding')
    eroded = erode(imageThresholding, 3)
    print('Erode - HistThresholding')
    print(erode)
    opened = dilate(eroded, 1)
    print('Dilate - HistThresholding')

    return opened

def area(image: np.array) -> int:

    unique, counts = np.unique(image, return_counts=True)
    counter = dict(zip(unique, counts))

    blackPixels = counter[0]
    print('unique %s counts %s', unique, counts)
    print('blackPixels %s', blackPixels)
    return blackPixels

def entropy(image: np.array, hist: np.array) -> int:
    length = sum(hist)
    probability = [float(h) / length for h in hist]
    entropy = -sum([p * math.log(p, 2) for p in probability if p != 0])
    return entropy


def calculateBoundRadius(segmented_img: np.array) -> float:

    center = np.array((0.0, 0.0))  # Fake init radius

    radius = 0.0001  # Fake init radius

    for _ in range(2):
        for pos, x in np.ndenumerate(segmented_img):

            arr_pos = np.array(pos)

            if x != 0:
                continue

            diff = arr_pos - center
            dist = np.sqrt(np.sum(diff ** 2))

            if dist < radius:
                continue

            alpha = dist / radius
            alphaSq = alpha ** 2

            radius = 0.5 * (alpha + 1.0 / alpha) * radius

            center = 0.5 * (
                (1.0 + 1.0 / alphaSq) * center + (1.0 - 1.0 / alphaSq) * arr_pos
            )

    for idx, _ in np.ndenumerate(segmented_img):

        arr_pos = np.array(idx)

        diff = arr_pos - center
        dist = np.sqrt(np.sum(diff ** 2))

        if dist < radius:
            break

        radius = (radius + dist) / 2.0
        center += (dist - radius) / dist * np.subtract(arr_pos, center)

    return radius

def crossValidationSplit(featureDataSet: np.array, n_folds: int) -> np.array:
    resultFoldDataSet = []
    copyFeatureDataSet = featureDataSet.copy()
    fold_size = int(len(featureDataSet)) // n_folds
    for ind in range(n_folds):
        fold = []

        while len(fold) < fold_size:
            index = randrange(len(copyFeatureDataSet))
            fold.append(copyFeatureDataSet[index])
            copyFeatureDataSet = np.delete(copyFeatureDataSet, index, axis=0)

        resultFoldDataSet.append(fold)

    return np.array(resultFoldDataSet)

def euclideanDistance(row1: np.array, row2: np.array) -> float:
    distance = 0.0
    for i in range(len(row1)-1):
	    distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def getNeighbors(train: np.array, test_row: np.array, K: int) -> np.array:
    distances = [(train_row, euclideanDistance(test_row, train_row)) for train_row in train]
    distances.sort(key=lambda tup: tup[1])

    neighbors = np.array([distances[i][0] for i in range(K)])

    return neighbors

def makePrediction(train: np.array, test_row: np.array, K: int = 3) -> np.array:
    neighbors = getNeighbors(train, test_row, K)

    outputValues = [row[-1] for row in neighbors]

    prediction = max(set(outputValues), key=outputValues.count)

    return prediction

def kNearestNeighbors(train: np.array, test: np.array, K: int) -> np.array:
    return np.array([makePrediction(train, row, K) for row in test])

def getAccuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
	    if actual[i] == predicted[i]:
		    correct += 1
    return correct / float(len(actual)) * 100.0
    