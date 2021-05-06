# imports
import os
import toml
import numpy as np  
import numba as nb
import sys, traceback
from typing import List
import matplotlib.pyplot as plt
import time
import os
from colorama import Fore, Style
from PIL import Image # Used only for importing and exporting images
from imageLib import *

# featuresList = np.array([])
featuresListCsv = []

class Features:
    def __init__(self, entropyResultFeature1, areaCalculatedFeature2, histogramMeanFeature3, boundRadiusFeature4, labelFeature5) -> None:
        self.entropyResultFeature1 = entropyResultFeature1
        self.areaCalculatedFeature2 = areaCalculatedFeature2
        self.histogramMeanFeature3 = histogramMeanFeature3
        self.boundRadiusFeature4 = boundRadiusFeature4
        self.labelFeature5 = labelFeature5

def groupImageLabel(entries):
    """Group images into class based on their name

    Args:
        entries ([type]): Batch files in directory
    """
    imgLabel = ""

    for entry in entries:
        if entry.is_file():
            if entry.name.find('cyl') != -1:
                imgLabel = 1
            
            if entry.name.find('inter') != -1:
                imgLabel = 2
            
            if entry.name.find('para') != -1:
                imgLabel = 3
            
            if entry.name.find('super') != -1:
                imgLabel = 4

            if entry.name.find('let') != -1:
                imgLabel = 5

            if entry.name.find('mod') != -1:
                imgLabel = 6

            if entry.name.find('svar') != -1:
                imgLabel = 7
    
            # processImageFeatures(entry, imgLabel)
    
    """ print('Completed Processing List')
    print(featuresListCsv)
    np.savetxt(conf["CSV_FILE"], featuresListCsv, delimiter=',') """
    print('KNN!')
    knn()

def knn():
    try:
        print('Test')
        print(conf["K_MAX_BOUND"])
        featuresDataSet = np.loadtxt(conf["CSV_FILE"], delimiter=",")
        # print(featuresDataSet)
        scores = []
        K = int(conf["K_MAX_BOUND"]) 
        totalAverage = 0
        for k in range(1, K + 1):
            print("Running Experiment k = {0}".format(k))
            # 10 fold cross validation
            folds = crossValidationSplit(featuresDataSet, conf["FOLDS"])
            # print(folds)

            for index,fold in enumerate(folds):
                test_dataset = fold
                copy_folds = np.copy(folds)
                train_dataset = np.concatenate(np.delete(copy_folds, index, axis=0), axis = 0)
                
                actual_class_column = np.size(test_dataset,1) - 1
                actual = test_dataset[:,actual_class_column]
                # actual = [row[-1] for row in fold]
                prediction = kNearestNeighbors(train_dataset, test_dataset, k)
                accuracy = getAccuracy(actual, prediction)
                scores.append(accuracy)

            print('Scores - {0}'.format(scores))
            print('Mean Accuracy - {0}'.format(accuracy))
        
        averageOfScores = sum(scores) / len(scores)
        totalAverage += averageOfScores

        finalTotalAvg = totalAverage / K
                
        print('Final Total Avg {0}'.format(finalTotalAvg))
        print('******END**********')
        
    except Exception as e:
        print('Error %s', e)
        traceback.print_exc()
        return e

def processImageFeatures(entry, imgLabel: int):
    """
    1) Process all the segmentation images from input folder and generate features
    2) Save the generated features array to features.csv file

    Args:
        entries ([type]): Batch files in directory
    """
    try:
        print('Image Label - %s', imgLabel)

        # Get image details
        segmentedImage = np.asarray(Image.open(conf["INPUT_DIR"] + entry.name))

        # Converting color images to selected single color spectrum
        singleSpectrumSegmentedImage = getSingleChannel(segmentedImage, conf["COLOR_CHANNEL"])

        # Histogram calculation for each individual image
        histogramResult = histogram(singleSpectrumSegmentedImage)

        # Histogram calculation for each individual image
        openResult = opening(singleSpectrumSegmentedImage, histogramResult)
        print(openResult)

        # Feature1 - Calculate Entropy
        entropyResultFeature1 = entropy(singleSpectrumSegmentedImage, histogramResult)
        print('entropyResultFeature1 %s', entropyResultFeature1)

        # Feature2 - Calculate Area
        areaCalculatedFeature2 = area(openResult)
        print('areaCalculatedFeature2 - %s', areaCalculatedFeature2)

        # Feature 3 - Calculate Mean
        histogramMeanFeature3 = np.mean(histogramResult)
        print('histogramMeanFeature3 %d', histogramMeanFeature3)

        # Feature 4 - Calcualte bound radius
        boundRadiusFeature4 = calculateBoundRadius(openResult)
        print('boundRadiusFeature4 %d', boundRadiusFeature4)

        # Feature 5 - Last column label name
        labelFeature5 = imgLabel
        print('labelFeature5 %d', labelFeature5)

        # Add features to list for each image
        addFeatureToList = np.array([entropyResultFeature1, areaCalculatedFeature2, histogramMeanFeature3, boundRadiusFeature4, labelFeature5])
        featuresListCsv.append(addFeatureToList)

    except Exception as e:
        print('Error %s', e)
        traceback.print_exc()
        return e

# Process files in directory as a batch
def process_batch(input):
    base_path = conf["INPUT_DIR"]
    with os.scandir(base_path) as entries:
        groupImageLabel(entries)

def main():
    print('----------IMAGE ANALYSIS START-------------------')
    print('Processing.......Results will be displayed after completion.....')
    global conf
    conf = toml.load('./config.toml')

    process_batch(conf)

if __name__ == "__main__":
    main()