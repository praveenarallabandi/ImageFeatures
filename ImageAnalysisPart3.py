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

total_start_time = time.time()

def groupImageLabel(entries):
    """Group images into class based on their name

    Args:
        entries ([type]): Batch files in directory
    """
    imgLabel = ""

    for entry in entries:
        if entry.is_file():
            if entry.name.find('cyl') != -1:
                imgLabel = 'cyl'
            
            if entry.name.find('inter') != -1:
                imgLabel = 'inter'
            
            if entry.name.find('para') != -1:
                imgLabel = 'para'
            
            if entry.name.find('super') != -1:
                imgLabel = 'super'

            if entry.name.find('let') != -1:
                imgLabel = 'let'

            if entry.name.find('mod') != -1:
                imgLabel = 'mod'

            if entry.name.find('svar') != -1:
                imgLabel = 'svar'
    
            processImageFeatures(entry, imgLabel)

def processImageFeatures(entry, imgLabel):
    features =[]
    try:
        print('Image Label - %s', imgLabel)

        # Get image details
        segmentedImage = np.asarray(Image.open(conf["INPUT_DIR"] + entry.name))

        # Converting color images to selected single color spectrum
        singleSpectrumSegmentedImage = getSingleChannel(segmentedImage, conf["COLOR_CHANNEL"])

        # Histogram calculation for each individual image
        histogramResult = histogram(singleSpectrumSegmentedImage)
        # print(histogramResult)
        # print(len(bins2))

        # Histogram - mean calculation
        histogramMean = np.mean(histogramResult)
        print('Mean %d', histogramMean)

        # Histogram calculation for each individual image
        openResult = opening(singleSpectrumSegmentedImage, histogramResult)
        print(openResult)

        # Calculate area
        areaCalculated = area(openResult)
        print('area - areaCalculated')
        print(areaCalculated)

        # Calculate entropy
        entropyResult = entropy(singleSpectrumSegmentedImage, histogramResult)
        print('entropyResult')
        print(entropyResult)

    except Exception as e:
        print('Error %s', e)
        traceback.print_exc()
        return e
    return features

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