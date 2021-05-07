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

featuresListCsv = []

def normalize(dataset: np.array) -> np.array:

    normalizedDataset = dataset.copy()
    imageLabels = dataset[:, :-1]
    for index, column in enumerate(imageLabels.T):
        smallest = np.min(column)
        largest = np.max(column)
        range = largest - smallest
        if range == 0:
            continue
        normalizedDataset[:, index] = (normalizedDataset[:, index] - smallest) / range
    return normalizedDataset

def groupImageClass(entries):
    """Image class based in their name

    Args:
        entries ([type]): Batch files in directory
    """
    imgLabel = ""
    classLabelName = ""

    for entry in entries:
        if entry.is_file() and entry.name != '.DS_Store':
            if entry.name.find('cyl') != -1:
                imgLabel = 1
                classLabelName = 'cyl'
            
            if entry.name.find('inter') != -1:
                imgLabel = 2
                classLabelName = 'inter'
            
            if entry.name.find('para') != -1:
                imgLabel = 3
                classLabelName = 'para'
            
            if entry.name.find('super') != -1:
                imgLabel = 4
                classLabelName = 'super'

            if entry.name.find('let') != -1:
                imgLabel = 5
                classLabelName = 'let'

            if entry.name.find('mod') != -1:
                imgLabel = 6
                classLabelName = 'mod'

            if entry.name.find('svar') != -1:
                imgLabel = 7
                classLabelName = 'svar'
    
            processImageFeatures(entry, imgLabel, classLabelName)
    
    # Save features to CSV
    normalizeRes = normalize(np.array(featuresListCsv))
    np.savetxt(conf["CSV_FILE"], np.array(normalizeRes), delimiter=',')
    print('----------Processing Features - completed-----------')
    print('Results saved to featue.csv file')
    print('--------------------KNN Started---------------------')
    knn()

def knn():
    try:
        featuresDataSet = np.loadtxt(conf["CSV_FILE"], delimiter=",")
        # print(featuresDataSet)
        K = int(conf["K_MAX_BOUND"]) 
        totalAverageAccuracy = 0.0
        for k in range(1, K + 1):
            print("Running Experiment k = {0}".format(k))
            # 10 fold cross validation
            folds = crossValidationSplit(featuresDataSet, conf["FOLDS"])
            scores = []
            avgAccuracy = 0.0
            for index,fold in enumerate(folds):
                avgAccuracy = 0.0
                test_dataset = fold
                copy_folds = np.copy(folds)
                train_dataset = np.concatenate(np.delete(copy_folds, index, axis=0), axis = 0)
                
                actualClassColumn = np.size(test_dataset,1) - 1
                actual = test_dataset[:,actualClassColumn]
                predicted = kNearestNeighbors(train_dataset, test_dataset, k)
                accuracy = getAccuracy(actual, predicted)
                scores.append(accuracy)
                avgAccuracy = sum(scores) / float(len(scores))
            print("avgAccuracy {0}".format(avgAccuracy))
            totalAverageAccuracy += avgAccuracy
            print(f"\tScores: {['{:.3f}%'.format(score) for score in scores]}")
            print(f"\tMean Accuracy: {avgAccuracy:.3f}%")
        
        finalTotalAvg = totalAverageAccuracy / K
                
        print(f"Final Average Accuracy : {finalTotalAvg:.3f}%")
        print('************END**********')
        
    except Exception as e:
        print('Error %s', e)
        traceback.print_exc()
        return e

def processImageFeatures(entry, imgLabel: int, classLabelName):
    """
    1) Process all the segmentation images from input folder and generate features
    2) Save the generated features array to features.csv file

    Args:
        entries ([type]): Batch files in directory
    """
    try:
        print('Processing Image: {0}'.format(entry.name))

        # Get image details
        image = np.asarray(Image.open(conf["INPUT_DIR"] + entry.name))
        
        singleSpectrumImage = getSingleChannel(image, conf["COLOR_CHANNEL"])

        # Histogram calculation for each individual image
        histogramResult = histogram(singleSpectrumImage)
        # segmentation
        openingResult = opening(singleSpectrumImage, histogramResult)

        # Feature1 - Calculate Entropy
        entropyResultFeature1 = entropy(singleSpectrumImage, histogramResult)
        print('entropyResultFeature1: {0}'.format(entropyResultFeature1))

        # Feature2 - Calculate Area
        areaCalculatedFeature2 = area(openingResult)
        print('areaCalculatedFeature2: {0}'.format(areaCalculatedFeature2))

        # Feature 3 - Calculate Mean
        histogramMeanFeature3 = np.mean(histogramResult)
        print('histogramMeanFeature3: {0}'.format(histogramMeanFeature3))

        # Feature 4 - Calculate radius
        boundRadiusFeature4 = calculateBoundRadius(openingResult)
        print('boundRadiusFeature4: {0}'.format(boundRadiusFeature4))

        # Feature 5 - Last column label name
        labelFeature5 = imgLabel
        print('labelFeature5: {0} - {1}'.format(labelFeature5, classLabelName))

        # Add features to list for each image
        addFeatureToList = np.array([entropyResultFeature1, areaCalculatedFeature2, histogramMeanFeature3, boundRadiusFeature4, labelFeature5])
        featuresListCsv.append(addFeatureToList)

        print('Processing Image: {0} - Done!'.format(entry.name))

    except Exception as e:
        print('Error %s', e)
        traceback.print_exc()
        return e

# Process files in directory as a batch
def process_batch(input):
    base_path = conf["INPUT_DIR"]
    with os.scandir(base_path) as entries:
        groupImageClass(entries)

def main():
    print('----------IMAGE ANALYSIS START-------------------')
    print('Processing Features...........')
    global conf
    conf = toml.load('./config.toml')

    process_batch(conf)

if __name__ == "__main__":
    main()