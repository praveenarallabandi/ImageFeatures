import numpy as np # Import Numpy library
 
# File name: five_fold_stratified_cv.py
# Author: Addison Sears-Collins
# Date created: 6/20/2019
# Python version: 3.7
# Description: Implementation of five-fold stratified cross-validation
# Divide the data set into five random groups. Make sure 
# that the proportion of each class in each group is roughly equal to its 
# proportion in the entire data set.
 
# Required Data Set Format for Disrete Class Values
# Classification:
# Must be all numerical
# Columns (0 through N)
# 0: Instance ID
# 1: Attribute 1 
# 2: Attribute 2
# 3: Attribute 3 
# ...
# N: Actual Class
 
# Required Data Set Format for Continuous Class Values:
# Regression:
# Must be all numerical
# Columns (0 through N)
# 0: Instance ID
# 1: Attribute 1 
# 2: Attribute 2
# 3: Attribute 3 
# ...
# N: Actual Class
# N + 1: Stratification Bin
 
class FiveFoldStratCv:
 
    # Constructor
    # Parameters: 
    #   np_dataset: The entire original data set as a numpy array
    #   problem_type: 'r' for regression and 'c' for classification 
    def __init__(self, np_dataset, problem_type):
        self.__np_dataset = np_dataset
        self.__problem_type = problem_type
   
    # Returns: 
    #   fold0, fold1, fold2, fold3, fold4
    #   Five folds whose class frequency distributions are 
    #   each representative of the entire original data set (i.e. Five-Fold 
    #   Stratified Cross Validation)
    def get_five_folds(self):
 
        # Record the number of columns in the data set
        no_of_columns = np.size(self.__np_dataset,1)
 
        # Record the number of rows in the data set
        no_of_rows = np.size(self.__np_dataset,0)
 
        # Create five empty folds (i.e. numpy arrays: fold0 through fold4)
        fold0 = np.arange(1)
        fold1 = np.arange(1)
        fold2 = np.arange(1)
        fold3 = np.arange(1)
        fold4 = np.arange(1)
 
        # Shuffle the data set randomly
        np.random.shuffle(self.__np_dataset)
 
        # Generate folds for classification problem
        if self.__problem_type == "c":
 
            # Record the column of the Actual Class
            actual_class_column = no_of_columns - 1
 
            # Generate an array containing the unique 
            # Actual Class values
            unique_class_arr = np.unique(self.__np_dataset[
                :,actual_class_column])
 
            unique_class_arr_size = unique_class_arr.size
 
            # For each unique class in the unique Actual Class array
            for unique_class_arr_idx in range(0, unique_class_arr_size):
 
                # Initialize the counter to 0
                counter = 0
 
                # Go through each row of the data set and find instances that
                # are part of this unique class. Distribute them among one
                # of five folds
                for row in range(0, no_of_rows):
 
                    # If the value of the unique class is equal to the actual
                    # class in the original data set on this row
                    if unique_class_arr[unique_class_arr_idx] == (
                        self.__np_dataset[row,actual_class_column]):
 
                            # Allocate instance to fold0
                            if counter == 0:
 
                                # If fold has not yet been created
                                if np.size(fold0) == 1:
 
                                    fold0 = self.__np_dataset[row,:]
 
                                    # Increase the counter by 1
                                    counter += 1
 
                                # Append this instance to the fold
                                else:
 
                                    # Extract data for the new row
                                    new_row = self.__np_dataset[row,:]
 
                                    # Append that entire instance to fold
                                    fold0 = np.vstack([fold0,new_row])
                                     
                                    # Increase the counter by 1
                                    counter += 1
 
                            # Allocate instance to fold1
                            elif counter == 1:
 
                                # If fold has not yet been created
                                if np.size(fold1) == 1:
 
                                    fold1 = self.__np_dataset[row,:]
 
                                    # Increase the counter by 1
                                    counter += 1
 
                                # Append this instance to the fold
                                else:
 
                                    # Extract data for the new row
                                    new_row = self.__np_dataset[row,:]
 
                                    # Append that entire instance to fold
                                    fold1 = np.vstack([fold1,new_row])
                                     
                                    # Increase the counter by 1
                                    counter += 1
 
                            # Allocate instance to fold2
                            elif counter == 2:
 
                                # If fold has not yet been created
                                if np.size(fold2) == 1:
 
                                    fold2 = self.__np_dataset[row,:]
 
                                    # Increase the counter by 1
                                    counter += 1
 
                                # Append this instance to the fold
                                else:
 
                                    # Extract data for the new row
                                    new_row = self.__np_dataset[row,:]
 
                                    # Append that entire instance to fold
                                    fold2 = np.vstack([fold2,new_row])
                                     
                                    # Increase the counter by 1
                                    counter += 1
 
                            # Allocate instance to fold3
                            elif counter == 3:
 
                                # If fold has not yet been created
                                if np.size(fold3) == 1:
 
                                    fold3 = self.__np_dataset[row,:]
 
                                    # Increase the counter by 1
                                    counter += 1
 
                                # Append this instance to the fold
                                else:
 
                                    # Extract data for the new row
                                    new_row = self.__np_dataset[row,:]
 
                                    # Append that entire instance to fold
                                    fold3 = np.vstack([fold3,new_row])
                                     
                                    # Increase the counter by 1
                                    counter += 1
 
                            # Allocate instance to fold4
                            else:
 
                                # If fold has not yet been created
                                if np.size(fold4) == 1:
 
                                    fold4 = self.__np_dataset[row,:]
 
                                    # Reset counter to 0
                                    counter = 0
 
                                # Append this instance to the fold
                                else:
 
                                    # Extract data for the new row
                                    new_row = self.__np_dataset[row,:]
 
                                    # Append that entire instance to fold
                                    fold4 = np.vstack([fold4,new_row])
                                     
                                    # Reset counter to 0
                                    counter = 0
 
        # If this is a regression problem
        else:
            # Record the column of the Stratification Bin
            strat_bin_column = no_of_columns - 1
 
            # Generate an array containing the unique 
            # Stratification Bin values
            unique_bin_arr = np.unique(self.__np_dataset[
                :,strat_bin_column])
 
            unique_bin_arr_size = unique_bin_arr.size
 
            # For each unique bin in the unique Stratification Bin array
            for unique_bin_arr_idx in range(0, unique_bin_arr_size):
 
                # Initialize the counter to 0
                counter = 0
 
                # Go through each row of the data set and find instances that
                # are part of this unique bin. Distribute them among one
                # of five folds
                for row in range(0, no_of_rows):
 
                    # If the value of the unique bin is equal to the actual
                    # bin in the original data set on this row
                    if unique_bin_arr[unique_bin_arr_idx] == (
                        self.__np_dataset[row,strat_bin_column]):
 
                            # Allocate instance to fold0
                            if counter == 0:
 
                                # If fold has not yet been created
                                if np.size(fold0) == 1:
 
                                    fold0 = self.__np_dataset[row,:]
 
                                    # Increase the counter by 1
                                    counter += 1
 
                                # Append this instance to the fold
                                else:
 
                                    # Extract data for the new row
                                    new_row = self.__np_dataset[row,:]
 
                                    # Append that entire instance to fold
                                    fold0 = np.vstack([fold0,new_row])
                                     
                                    # Increase the counter by 1
                                    counter += 1
 
                            # Allocate instance to fold1
                            elif counter == 1:
 
                                # If fold has not yet been created
                                if np.size(fold1) == 1:
 
                                    fold1 = self.__np_dataset[row,:]
 
                                    # Increase the counter by 1
                                    counter += 1
 
                                # Append this instance to the fold
                                else:
 
                                    # Extract data for the new row
                                    new_row = self.__np_dataset[row,:]
 
                                    # Append that entire instance to fold
                                    fold1 = np.vstack([fold1,new_row])
                                     
                                    # Increase the counter by 1
                                    counter += 1
 
                            # Allocate instance to fold2
                            elif counter == 2:
 
                                # If fold has not yet been created
                                if np.size(fold2) == 1:
 
                                    fold2 = self.__np_dataset[row,:]
 
                                    # Increase the counter by 1
                                    counter += 1
 
                                # Append this instance to the fold
                                else:
 
                                    # Extract data for the new row
                                    new_row = self.__np_dataset[row,:]
 
                                    # Append that entire instance to fold
                                    fold2 = np.vstack([fold2,new_row])
                                     
                                    # Increase the counter by 1
                                    counter += 1
 
                            # Allocate instance to fold3
                            elif counter == 3:
 
                                # If fold has not yet been created
                                if np.size(fold3) == 1:
 
                                    fold3 = self.__np_dataset[row,:]
 
                                    # Increase the counter by 1
                                    counter += 1
 
                                # Append this instance to the fold
                                else:
 
                                    # Extract data for the new row
                                    new_row = self.__np_dataset[row,:]
 
                                    # Append that entire instance to fold
                                    fold3 = np.vstack([fold3,new_row])
                                     
                                    # Increase the counter by 1
                                    counter += 1
 
                            # Allocate instance to fold4
                            else:
 
                                # If fold has not yet been created
                                if np.size(fold4) == 1:
 
                                    fold4 = self.__np_dataset[row,:]
 
                                    # Reset counter to 0
                                    counter = 0
 
                                # Append this instance to the fold
                                else:
 
                                    # Extract data for the new row
                                    new_row = self.__np_dataset[row,:]
 
                                    # Append that entire instance to fold
                                    fold4 = np.vstack([fold4,new_row])
                                     
                                    # Reset counter to 0
                                    counter = 0
         
        return fold0, fold1, fold2, fold3, fold4