import pandas as pd # Import Pandas library 
import numpy as np # Import Numpy library
from five_fold_stratified_cv import FiveFoldStratCv
from knn import Knn
 
# File name: knn_driver.py
# Author: Addison Sears-Collins
# Date created: 6/20/2019
# Python version: 3.7
# Description: Driver for the knn.py program 
# (K-Nearest Neighbors)
 
# Required Data Set Format for Classification Problems:
# Must be all numerical
# Columns (0 through N)
# 0: Instance ID
# 1: Attribute 1 
# 2: Attribute 2
# 3: Attribute 3 
# ...
# N: Actual Class
 
# Required Data Set Format for Regression Problems:
# Must be all numerical
# Columns (0 through N)
# 0: Instance ID
# 1: Attribute 1 
# 2: Attribute 2
# 3: Attribute 3 
# ...
# N: Actual Class
# N + 1: Stratification Bin
 
################# INPUT YOUR OWN VALUES IN THIS SECTION ######################
ALGORITHM_NAME = "K-Nearest Neighbor"
SEPARATOR = ","  # Separator for the data set (e.g. "\t" for tab data)
##############################################################################
 
def main():
 
    print("Welcome to the " +  ALGORITHM_NAME + " program!")
    print()
    print("Running " + ALGORITHM_NAME + ". Please wait...")
    print()
 
    # k value that will be tuned
    k = eval(input("Enter a value for k: ") )
 
    # "c" for classification or "r" for regression
    problem = input("Press  \"c\" for classification or \"r\" for regression: ") 
 
    # Directory where data set is located
    data_path = input("Enter the path to your input file: ") 
 
    # Read the full text file and store records in a Pandas dataframe
    pd_data_set = pd.read_csv(data_path, sep=SEPARATOR)
 
    # Convert dataframe into a Numpy array
    np_data_set = pd_data_set.to_numpy(copy=True)
 
    # Show functioning of the program
    trace_runs_file = input("Enter the name of your trace runs file: ") 
 
    # Open a new file to save trace runs
    outfile_tr = open(trace_runs_file,"w") 
 
    # Testing statistics
    test_stats_file = input("Enter the name of your test statistics file: ") 
 
    # Open a test_stats_file 
    outfile_ts = open(test_stats_file,"w")
 
    # Create an object of class FiveFoldStratCv
    fivefolds1 = FiveFoldStratCv(np_data_set,problem)
 
    # The number of folds in the cross-validation
    NO_OF_FOLDS = 5
 
    # Create an object of class Knn
    knn1 = Knn(k,problem) 
 
    # Generate the five stratified folds
    fold0, fold1, fold2, fold3, fold4 = fivefolds1.get_five_folds()
 
    training_dataset = None
    test_dataset = None
 
    # Create an empty array of length 5 to store the accuracy_statistics 
    # (classification accuracy for classification problems or mean squared
    # error for regression problems)
    accuracy_statistics = np.zeros(NO_OF_FOLDS)
 
    # Run k-NN the designated number of times as indicated by the number of folds
    for experiment in range(0, NO_OF_FOLDS):
 
        print()
        print("Running Experiment " + str(experiment + 1) + " ...")
        print()
        outfile_tr.write("Running Experiment " + str(experiment + 1) + " ...\n")
        outfile_tr.write("\n")
 
        # Each fold will have a chance to be the test data set
        if experiment == 0:
            test_dataset = fold0
            training_dataset = np.concatenate((
                fold1, fold2, fold3, fold4), axis=0)
        elif experiment == 1:
            test_dataset = fold1
            training_dataset = np.concatenate((
                fold0, fold2, fold3, fold4), axis=0)
        elif experiment == 2:
            test_dataset = fold2
            training_dataset = np.concatenate((
                fold0, fold1, fold3, fold4), axis=0)
        elif experiment == 3:
            test_dataset = fold3
            training_dataset = np.concatenate((
                fold0, fold1, fold2, fold4), axis=0)
        else:
            test_dataset = fold4
            training_dataset = np.concatenate((
                fold0, fold1, fold2, fold3), axis=0)
 
        # Actual class column index of the test dataset
        actual_class_column = None          
      
        # If classification problem
        if problem == "c":
            actual_class_column = np.size(test_dataset,1) - 1
        # If regression problem
        else:
            actual_class_column = np.size(test_dataset,1) - 2
 
        # Create an array of the actual_class_values of the test instances
        actual_class_values = test_dataset[:,actual_class_column]
 
        no_of_test_instances = np.size(test_dataset,0)
 
        # Make an empty array called predicted_class_values which will 
        # store the predicted class values. It should be the same length 
        # as the number of test instances
        predicted_class_values = np.zeros(no_of_test_instances)
 
        # For each row in the test data set
        for row in range(0, no_of_test_instances):
   
            # Neighbor array is a 2D array containing the neighbors 
            # (actual class value, distance) 
            # Get the k nearest neighbors for each test instance
            this_instance = test_dataset[row,:]
            neighbor_array = knn1.get_neighbors(training_dataset,this_instance)
 
            # Extract the actual class values
            neighbors_arr = neighbor_array[:,0]
   
            # Predicted class value stored in the variable prediction
            prediction = knn1.make_prediction(neighbors_arr)
   
            # Record the prediction in the predicted_class_values array
            predicted_class_values[row] = prediction
 
        # Calculate the classification accuracy of the predictions 
        # (k-NN classification) or the mean squared error (k-NN regression)
        accuracy = knn1.get_accuracy(actual_class_values,predicted_class_values)
 
        # Store the accuracy in the accuracy_statistics array
        accuracy_statistics[experiment] = accuracy
 
        # If classification problem
        if problem == "c":
            temp_acc = accuracy * 100
            outfile_tr.write("Classification Accuracy: " + str(temp_acc) + "%\n")
            outfile_tr.write("\n")
            print("Classification Accuracy: " + str(temp_acc) + "%\n")
        # If regression problem
        else:
            outfile_tr.write("Mean Squared Error: " + str(accuracy) + "\n")
            outfile_tr.write("\n")
            print("Mean Squared Error: " + str(accuracy) + "\n")
 
    outfile_tr.write("Experiments Completed.\n")
    print("Experiments Completed.")
    print()
 
    # Write to a file
    outfile_ts.write("----------------------------------------------------------\n")
    outfile_ts.write(ALGORITHM_NAME + " Summary Statistics\n")
    outfile_ts.write("----------------------------------------------------------\n")
    outfile_ts.write("Data Set : " + data_path + "\n")
    outfile_ts.write("\n")
    outfile_ts.write("Accuracy Statistics for All 5 Experiments:")
    outfile_ts.write(np.array2string(
        accuracy_statistics, precision=2, separator=',',
        suppress_small=True))
    outfile_ts.write("\n")
 
    # Write the relevant stats to a file
    outfile_ts.write("\n")
 
    if problem == "c":
        outfile_ts.write("Problem Type : Classification" + "\n")
    else:
        outfile_ts.write("Problem Type : Regression" + "\n")
 
    outfile_ts.write("\n")
    outfile_ts.write("Value for k : " + str(k) + "\n")
    outfile_ts.write("\n")
 
    accuracy = np.mean(accuracy_statistics)
 
    if problem == "c":
        accuracy *= 100
        outfile_ts.write("Classification Accuracy : " + str(accuracy) + "%\n")
    else: 
        outfile_ts.write("Mean Squared Error : " + str(accuracy) + "\n")
 
    # Print to the console
    print()
    print("----------------------------------------------------------")
    print(ALGORITHM_NAME + " Summary Statistics")
    print("----------------------------------------------------------")
    print("Data Set : " + data_path)
    print()
    print()
    print("Accuracy Statistics for All 5 Experiments:")
    print(accuracy_statistics)
    print()
    # Write the relevant stats to a file
    print()
 
    if problem == "c":
        print("Problem Type : Classification")
    else:
        print("Problem Type : Regression")
 
    print()
    print("Value for k : " + str(k))
    print()
    if problem == "c":
        print("Classification Accuracy : " + str(accuracy) + "%")
    else: 
        print("Mean Squared Error : " + str(accuracy))
 
    print()
 
    # Close the files
    outfile_tr.close()
    outfile_ts.close()
 
main()