# imports
import math
import numpy as np
import numba as nb
from operator import eq
from typing import List
import matplotlib.pyplot as plt
from colorama import Fore, Style
from PIL import Image  # Used only for importing and exporting images
from random import randrange
from math import sqrt


def getSingleChannel(image: np.array, colorSpectrum: str) -> np.array:
    if(image.ndim == 3):
        if(colorSpectrum == 'R'):
            img = image[:, :, 0]

        if(colorSpectrum == 'G'):
            img = image[:, :, 1]

        if(colorSpectrum == 'B'):
            img = image[:, :, 2]

        return img
    else:
        return image


def histogram(image: np.array) -> np.array:
    imageSize: int = len(image)
    hist: np.array = np.zeros(256)
    for pixel in range(256):
        for index in range(imageSize):
            if image.flat[index] == pixel:
                hist[pixel] += 1
    return hist

def histogramThresholding(image: np.array, hist: np.array) -> np.array:
    if hist.any() == None:
        hist = histogram(image)

    bins = len(hist)
    histogramStart = 0
    while hist[histogramStart] < 5:
        histogramStart += 1

    histogramEnd = bins - 1
    while hist[histogramEnd] < 5:
        histogramEnd -= 1

    maxVal = 255.0
    histogramCenter = int(
        round(np.average(np.linspace(0, maxVal, bins), weights=hist)))
    left = np.sum(hist[histogramStart:histogramCenter])
    right = np.sum(hist[histogramCenter: histogramEnd + 1])

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

    imageCopy = image.copy()
    imageCopy[imageCopy > histogramCenter] = 255
    imageCopy[imageCopy < histogramCenter] = 0
    imageCopy = imageCopy.astype(np.uint8)
    return imageCopy.reshape(image.shape)

def getTileArray(array, n):
    return np.tile(array, (n, 1)).flatten()

def erosion(image: np.array, erosionLevel: int = 3) -> np.array:
    initArray = np.zeros_like(image)
    [y, x] = np.where(image > 0)

    tileArray = np.tile(range(-erosionLevel, erosionLevel + 1), (2 * erosionLevel + 1, 1))
    xTileArray = tileArray.flatten()
    yTileArray = tileArray.T.flatten()

    n = len(x.flatten())
    xTileArray = getTileArray(xTileArray, n)
    yTileArray = getTileArray(yTileArray, n)

    index = np.sqrt(xTileArray ** 2 + yTileArray ** 2) > erosionLevel
    xTileArray[index] = 0
    yTileArray[index] = 0

    reps = ((2 * erosionLevel + 1) ** 2)
    x = np.tile(x, reps)
    y = np.tile(y, reps)

    new_x = x + xTileArray
    new_y = y + yTileArray

    new_x[new_x < 0] = 0
    new_x[new_x > image.shape[1] - 1] = image.shape[1] - 1
    new_y[new_y < 0] = 0
    new_y[new_y > image.shape[0] - 1] = image.shape[0] - 1

    initArray[new_y, new_x] = 255
    imageErosion = initArray.astype(np.uint8)
    return imageErosion

def dilate(image: np.array, window: int = 1) -> np.array:
    invertImage = np.invert(image)
    erodedImage = erosion(invertImage, window).astype(np.uint8)
    eroded_img = np.invert(erodedImage)

    return eroded_img


def opening(image: np.array, histogram: np.array) -> np.array:
    imageThresholding = histogramThresholding(image, histogram)
    eroded = erosion(imageThresholding, 3)
    opened = dilate(eroded, 1)
    return opened

# Features


def area(image: np.array) -> int:
    """Calculate area

    Args:
        image (np.array): [description]

    Returns:
        int: area
    """
    unique, counts = np.unique(image, return_counts=True)
    counter = dict(zip(unique, counts))
    blackPixels = counter[0]
    return blackPixels


def entropy(image: np.array, hist: np.array) -> int:
    """Calculate entropy

    Args:
        image (np.array): [description]
        hist (np.array): [description]

    Returns:
        int: entropy
    """
    length = sum(hist)
    probability = [float(h) / length for h in hist]
    entropy = -sum([p * math.log(p, 2) for p in probability if p != 0])
    return entropy


def calculatePerimeter(image: np.array) -> float:
    interior = abs(np.diff(image, axis=0)).sum() + \
        abs(np.diff(image, axis=1)).sum()
    boundary = image[0, :].sum() + image[:, 0].sum() + \
        image[-1, :].sum() + image[:, -1].sum()
    perimeter = interior + boundary
    return perimeter

def crossValidationSplit(featureDataSet: np.array, n_folds: int) -> np.array:
    """10 fold crossvalidation

    Args:
        featureDataSet (np.array): [description]
        n_folds (int): folds from config.toml

    Returns:
        np.array: result array dataset
    """
    resultFoldDataSet = []
    copyFeatureDataSet = featureDataSet.copy()
    foldSize = int(len(featureDataSet)) // n_folds
    for ind in range(n_folds):
        fold = []

        while len(fold) < foldSize:
            index = randrange(len(copyFeatureDataSet))
            fold.append(copyFeatureDataSet[index])
            copyFeatureDataSet = np.delete(copyFeatureDataSet, index, axis=0)

        resultFoldDataSet.append(fold)

    return np.array(resultFoldDataSet)


def euclideanDistance(row1: np.array, row2: np.array) -> float:
    """Euclidean distance calculation

    Args:
        row1 (np.array): [description]
        row2 (np.array): [description]

    Returns:
        float: euclidean distance
    """
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


def getNeighbors(train: np.array, testRow: np.array, K: int) -> np.array:
    distances = [(train_row, euclideanDistance(testRow, train_row))
                 for train_row in train]
    distances.sort(key=lambda tup: tup[1])
    neighbors = np.array([distances[index][0] for index in range(K)])
    return neighbors


def makePrediction(train: np.array, testRow: np.array, K: int = 3) -> np.array:
    neighborsResult = getNeighbors(train, testRow, K)
    outputValues = [eachRow[-1] for eachRow in neighborsResult]
    prediction = max(set(outputValues), key=outputValues.count)
    return prediction


def kNearestNeighbors(train: np.array, testDataset: np.array, K: int) -> np.array:
    """KNN

    Args:
        train (np.array): [description]
        testDataset (np.array): [description]
        K (int): [description]

    Returns:
        np.array: Predicted Array
    """
    return np.array([makePrediction(train, row, K) for row in testDataset])


def getAccuracy(actual, predicted):
    """Calculate accuracy

    Args:
        actual ([type]): [description]
        predicted ([type]): [description]

    Returns:
        [type]: accuracy
    """
    correct = list(map(eq, actual, predicted))
    return (sum(correct) / len(correct)) * 100.0
