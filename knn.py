import numpy as np  

class Knn:
 
    # Constructor
    #   Parameters:
    #     k: k value for k-NN
    #     problem_type: ('r' for regression or 'c' for classification)
    def __init__(self, k, problem_type):
        self.__k = k
        self.__problem_type = problem_type
 
    # Parameters:
    #   training_set: The folds used for training (2d numpy array)
    #   test_instance: The test instance we need to find the neighbors for 
    #                  (numpy array)
    # Returns: 
    #   The k most similar neighbors from the training set for a given 
    #   test instance. It will be a 2D numpy array where the first column 
    #   will hold the Actual class value of the training instance and the 
    #   second column will store the distance to the test instance...
    #   (actual class value, distance). 
    def get_neighbors(self, training_set, test_instance):
 
        # Record the number of training instances in the training set
        no_of_training_instances = np.size(training_set,0)
 
        # Record the number of columns in the training set
        no_of_training_columns = np.size(training_set,1)
 
        # Record the column index of the actual class of the training_set
        actual_class_column = None
        # If classification problem
        if self.__problem_type == "c":
            actual_class_column = no_of_training_columns - 1
        # If regression problem
        else:
            actual_class_column = no_of_training_columns - 2
 
        # Create an empty 2D array called actual_class_and_distance. This 
        # array should be the same length as the number of training instances. 
        # The first column will hold the Actual Class value of the training 
        # instance, and the second column will store the distance to the 
        # test instance...(actual class value, distance). 
        actual_class_and_distance = np.zeros((no_of_training_instances, 2))
 
        neighbors = None
 
        # For each row (training instance) in the training set
        for row in range(0, no_of_training_instances):
 
            # Record the actual class value in the 
            # actual_class_and_distance array (column 0)
            actual_class_and_distance[row,0] = training_set[
                row,actual_class_column]
 
            # Initialize a temporary training instance copied from this 
            # training instance
            temp_training_instance = np.copy(training_set[row,:])
 
            # Initialize a temporary test instance copied from this 
            # test_instance
            temp_test_instance = np.copy(test_instance)
 
            # If this is a classification problem
            if self.__problem_type == "c":
             
                # Update temporary training instance with Instance ID 
                # and the Actual Class pieces removed
                temp_training_instance = np.delete(temp_training_instance,[
                    0,actual_class_column])
 
                # Update temporary test instance with the Instance ID 
                # and the Actual Class pieces removed
                temp_test_instance = np.delete(temp_test_instance,[
                    0,actual_class_column])
 
            # If this is a regression problem
            else:
 
                # Update temporary training instance with Instance ID, Actual 
                # Class, and Stratification Bin pieces removed
                temp_training_instance = np.delete(temp_training_instance,[
                    0,actual_class_column,actual_class_column+1])
 
                # Update temporary test instance with the Instance ID, Actual
                # Class, and Stratification Bin pieces removed
                temp_test_instance = np.delete(temp_test_instance,[
                    0,actual_class_column,actual_class_column+1])
 
            # Calculate the euclidean distance from the temporary test
            # instance to the temporary training instance
            distance = np.linalg.norm(
                temp_test_instance - temp_training_instance)
 
            # Record the distance in the actual_class_and_distance 
            # array (column 1)
            actual_class_and_distance[row,1] = distance
 
        # Sort the actual_class_and_distance 2D array by the 
        # distance column (column 1) in ascending order.
        actual_class_and_distance = actual_class_and_distance[
                actual_class_and_distance[:,1].argsort()]
 
        k = self.__k
 
        # Extract the first k rows of the actual_class_and_distance array
        neighbors = actual_class_and_distance[:k,:]
 
        return neighbors
 
    # Generate a prediction based on the most frequent class or averaged 
    # target variable value of the neighbors
    # Parameters: 
    #   neighbors - 1D array (actual_class_value)
    # Returns:
    #   predicted_class_value
    def make_prediction(self, neighbors):
     
        prediction = None
 
        # If this is a classification problem
        if self.__problem_type == "c":
             
            #  Prediction is the most frequent value in column 0 of
            #  the neighbors array
            neighborsint = neighbors.astype(int)
            prediction = np.bincount(neighborsint).argmax()
             
        # If this is a regression problem
        else:
 
            # Prediction is the average of the neighbors array
            prediction = np.mean(neighbors)
 
        return prediction
 
    # Parameters: 
    #   actual_class_array
    #   predicted_class_array
    # Returns: 
    #   accuracy: Either classification accuracy (for 
    #   classification problems) or 
    #   mean squared error (for regression problems)
    def get_accuracy(self, actual_class_array, predicted_class_array):
 
        # Initialize accuracy variable
        accuracy = None
 
        # Initialize decision variable
        decision = None
 
        counter = None
 
        actual_class_array_size = actual_class_array.size
 
        # If this is a classification problem
        if self.__problem_type == "c":
             
            counter = 0
 
            # For each element in the actual class array
            for row in range(0,actual_class_array_size):
 
                # If actual class value is equal to value in predicted
                # class array
                if actual_class_array[row] == predicted_class_array[row]:
                    decision = "correct"
                    counter += 1
                else:
                    decision = "incorrect"
 
            classification_accuracy = counter / (actual_class_array_size)
 
            accuracy = classification_accuracy
 
        # If this is a regression problem
        else:
 
            # Initialize an empty squared error array. 
            # Needs to be same length as number of test instances
            squared_error_array = np.empty(actual_class_array_size)
 
            squared_error = None
 
            # For each element in the actual class array
            for row in range(0,actual_class_array_size):
 
                # Calculate the squared error
                squared_error = (abs((actual_class_array[
                        row] - predicted_class_array[row]))) 
                squared_error *= squared_error
                squared_error_array[row] = squared_error
 
            mean_squared_error = np.mean(squared_error_array)
             
            accuracy = mean_squared_error
         
        return accuracy