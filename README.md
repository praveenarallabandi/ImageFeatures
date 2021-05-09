# Image Analysis - Part1

## Overview


The objective of the program is to classify the images to their cell-appropriate types (cyl, inter, let, mod, para, super, or svar). Below are the series of steps performed on the images 
* Take cell images as input. 
* Extract Features and save them to CSV file
* Load the CSV feature file data for k nearest neighbor
* Execute k-nearest neighbor on the loaded dataset with k-fold cross-validation, outputting the accuracy of cross-validation
* A TOML file for configuration of attributes input, output, k-bound, and number of folds

## Usage

```
git clone https://github.com/praveenarallabandi/ImageFeatures.git
cd ImageAnalysis
pip3 install --user pipenv
python ImageAnalysisPart3.py
```

## Implementation

### Feature Extraction 

The feature extracted from images are
* Area of cluster - The area of the image by calculating pixels of a cluster on the morphologically opened image
* Entropy of image - Probability of each image combined with a single scalar value
* Histogram Mean - Mean of the histogram on each of the occurrences of each pixel value
* Perimeter - The perimeter of the image by summation of the interior and external boundaries of the image 

The project implementation is done using Python. Using Python, we can rapidly develop and integrate each operation. Python's NumPy library, which allows for array operations. 
Certain image array operations are time-consuming, and those scenarios were addressed with optimizing NumPy arrays (using NumPy methods as much as possible) and with numba. Numba is an open-source JIT compiler that translates a subset of Python and NumPy code into fast machine code. Numba has a python function decorator for just-in-time compiling functions to machine code before executing. Using this decorator on functions that use heavy math and looping (i.e., filters and noise) provides significant speed increases with speeds similar to using lower-level compiled languages like C/C++ or Rust. For plotting histograms, Python's `matplotlib,` the relatively standard and robust plotting library, outputs plots to a file with the rest of the exported output images.

## Dependencies 

* numpy - For Array operations
* matplotlib - Plot
* numba - JIT for speed exectuion
* toml - Configuration settings
* PIL (Image) - Used only for importing and exporting images

 ## Functions

```python
def calculateHistogram(image: np.array) -> np.array:
```
`calculate_histogram` Generates the histogram, equalized histogram, and quantized image based on the equalized histogram

```python
def image_quantization_mse(image: np.array, imageQuant: np.array, imageName: str) -> float:
```
 `image_quantization_mse` Calculates mean square error for two input images

```python
def convertToSingleColorSpectrum(image: np.array, colorSpectrum: str) -> np.array:
```
`convertToSingleColorSpectrum` Generates the NumPy array for a single color spectrum.

```python
def corruptImageGaussian(image: np.array, strength: int) ->  np.array:
```
`corruptImageGaussian` Generates image with gaussian noise applied

```python
def corruptImageSaltAndPepper(image: np.array, strength: int) -> np.array:
```
`corruptImageSaltAndPepper` Generates image with salt and pepper noise applied

```python
def linearFilter(image, maskSize=9, weights = List[List[int]]) -> np.array:
```
`linearFilter` Receives a kernel or matrix of weights as a two-dimensional input list and applies that kernel to a copy of an image. The filter is then applied in loops through each pixel in the image and multiples the neighboring pixels' values by the kernel weights. The larger the kernel, the larger the pixel's neighborhood that affects the pixel. 

```python
def medianFilter(image, maskSize=9, weights = List[List[int]]):
```
`medianFilter` The median filter is applied to the input image, and each pixel is replaced with the median value of its neighbors. The current pixel value as well is included in the median calculation.

## Results

The output images are stored in output directory, mean square errors for each image is printed on stdout alsong with performance metrics
Below is th snapshop of output

<img width="592" alt="Screen Shot 2021-03-15 at 9 50 24 PM" src="https://user-images.githubusercontent.com/44982889/111244241-e537c680-85d8-11eb-9479-3109ef7965f1.png">
