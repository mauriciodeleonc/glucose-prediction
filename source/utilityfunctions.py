'''
    utilityfunctions.py
    Python library with many ML functions
    including linear regression, gradient
    descent, L2 norm, amongst others.

    Author: Mauricio A. De Leon Cardenas
    Email: mauricio.deleonc@udem.edu
    Institution: Universidad de Monterrey
    First created: Sun 29 Mar 2020
'''

import numpy as np
import pandas as pd
import math

def load_data_multivariate(file):
    ''' 
        Load data from a 
        comma-separated-value file

        INPUT: 
            file: from which de data will be uploaded
            type: string conditional to determine what to return 
                (values: training || testing)
        
        OUTPUT:
            x_data: numpy array with all values of x
            y: numpy array with all values of y
            mean: numpy array with the mean values of every x
            std: numpy array with the standard deviation values
                of every x
    '''
    try:
        data = pd.read_csv(file)

        x_data = pd.DataFrame.to_numpy(data.loc[:, data.columns != 'Outcome'])

        if 'Outcome' in data:
            y = pd.Series.to_numpy(data['Outcome'])
            y = np.atleast_2d(y)
            y = y.T

        mean = np.mean(x_data, axis = 0)
        std = np.std(x_data, axis = 0)

        return x_data, y, mean, std
    except IOError as e:
        print("The file was not loaded correctly, try with another filename")
        exit(1)

def split_data(x_data, y_data, perc_training):
    ''' 
        Method which splits the total dataset
        into two, one part goes to training and
        the rest goes to testing. The amount
        to be split is defined by perc_training.

        INPUT: 
            x_data: numpy array with the entire feature values
                of the dataset
            y_data: numpy array with the entire outcome values
                of the dataset
            perc_training: float value indicating the percentage
                to be used for training data
        
        OUTPUT:
            x_training: numpy array only with the percentage
                of values from the x_data array
            y_training: numpy array only with the percentage
                of values from the y_data array
            x_testing: numpy array only with the left percentage
                of values from the x_data array
            y_testing: numpy array only with the left percentage
                of values from the x_data array
    '''
    rows = x_data.shape[0]
    test_num = math.ceil(rows * perc_training)
    i = 0
    x_training = []
    y_training = []
    x_testing = []
    y_testing = []

    for x in x_data:
        if i < test_num:
            x_training.append(x)
            i = i + 1
        else:
            x_testing.append(x)

    x_training = np.array(x_training, dtype = np.float64)
    #np.savetxt('x_training.csv', x_training, delimiter=',')
    x_testing = np.array(x_testing, dtype = np.float64)
    #np.savetxt('x_testgin.csv', x_testing, delimiter=',')

    i = 0
    for y in y_data:
        if i < test_num:
            y_training.append(y)
            i = i + 1
        else:
            y_testing.append(y)

    y_training = np.array(y_training, dtype = np.float64)
    y_testing = np.array(y_testing, dtype = np.float64)

    return x_training, y_training, x_testing, y_testing

def scale_data(x, mean, std):
    ''' 
        Scale values from a given numpy array
        using the mean and standar deviation values
        from each respective x

        INPUT: 
            x: numpy array (could be training, testing or 
                cross-validation values)
            mean: numpy array with the mean values of every x
            std: numpy array with the standard deviation values
                of every x
        
        OUTPUT:
            x_scaled: numpy array with all the scaled values of x
    '''
    x_scaled = (x - mean) / std
    return x_scaled

def eval_hypothesis_function_multivariate(w, x):
    '''
        Function that evaluates if the x data and
        w params array can be multiplied using
        the sigmoid function (needed for logistic 
        regression)

        INPUT:
            w: numpy array with the values of the parameters
            x: numpy array with the scaled values of x

        OUTPUT:
            return the multiplication of both arrays run
            through the sigmoid function
    '''
    #print(1/(1+np.exp(-np.dot(x.T,w))))
    return 1/(1+np.exp(-np.dot(x.T,w)))

def compute_gradient_of_cost_function_multivariate(x, y, w):
    '''
        Function that evaluates if the x data and
        w params array can be multiplied

        INPUT:
            x: numpy array with the scaled values of x
            y: numpy array with the real results of the training data
            w: numpy array with the values of the parameters

        OUTPUT:
            gradient_of_cost_function: result of the subtraction of the
                hypothesis function multiplied by the x data array divided by
                the total amount of data
    '''
    N = x.shape[1]

    hypothesis_function = eval_hypothesis_function_multivariate(w, x)
    
    residual = np.subtract(hypothesis_function, y)
    gradient_of_cost_function = (np.dot(residual.T, x.T)/ N)
    

    return gradient_of_cost_function

def compute_L2_norm_multivariate(gradient_of_cost_function, x):
    '''
        Function that calulates the stopping criteria with which
        the gradient descent will use to stop

        INPUT:
            gradient_of_cost_function: result of the subtraction of the
                hypothesis function multiplied by the x data array divided by
                the total amount of data
            x: numpy array with the scaled values of x
    '''
    return np.linalg.norm(gradient_of_cost_function)

def gradient_descent_multivariate(x_training, y_training, w, stopping_criteria, learning_rate):
    '''
        Function that calculates the gradient descent to achieve
        the best values for the parameters that represent the data given.

        INPUTS:
            x_training: numpy array with the scaled values of x
            y_training: numpy array with the real results of the training data
            w: numpy array with the values of the parameters
            stopping_criteria: value which when evaluated bigger than the L2
                norm tell the function to stop
            learning_rate: value at which the gradient will "advance" (descent)

        OUTPUT:
            w: numpy array with the final values of the parameters
                that best represent the behaviour of the data
            i: counter with the total amount of iterations needed in
                order to achieve those best parameters
    '''
    L2_norm = 1

    ones = np.atleast_2d(np.ones(len(x_training))).T
    x = np.hstack((ones, x_training))
    x = x.T

    i = 0
    while L2_norm > stopping_criteria:
        gradient_of_cost_function_multivariate = compute_gradient_of_cost_function_multivariate(x, y_training, w)
        
        w = w - learning_rate * gradient_of_cost_function_multivariate.T
        L2_norm = compute_L2_norm_multivariate(gradient_of_cost_function_multivariate, x_training)
        i = i + 1
        
    return w, i

def get_confusion_matrix(y_testing, predicted_data):
    '''
        Function that creates the confusion matrix.
        For terms of computational complexity this
        is created using a unidimensional array
        for better indexing access.

        INPUT:
            y_testing: numpy array with the real outcome 
                values separated specifically for testing
            predicted_data: numpy array with the predicted
                estimated values obtained with the w parameters

        OUTPUT:
            returns the confusion matrix (unidimensional array)

    '''
    confusion_matrix = [0,0,0,0]
    for y, pred in zip(y_testing, predicted_data):
        if y == 1 and pred == 1:
            confusion_matrix[0] = confusion_matrix[0] + 1
        elif y == 0 and pred == 1:
            confusion_matrix[1] = confusion_matrix[1] + 1
        elif y == 1 and pred == 0:
            confusion_matrix[2] = confusion_matrix[2] + 1
        elif y == 0 and pred == 0:  
            confusion_matrix[3] = confusion_matrix[3] + 1

    return confusion_matrix;

def print_performance_metrics(confusion_matrix):
    '''
        Function that obtains and prints the performance
        metrics based on the values obtaines from the
        confusion matrix.

        INPUT:
            confusion_matrix: unidimensional array where each
                cell indicates the relation between actual
                class and predicted class
        
        OUTPUT:
            returns None (it prints everything in the method)
    '''
    TP = confusion_matrix[0]
    FP = confusion_matrix[1]
    FN = confusion_matrix[2]
    TN = confusion_matrix[3]
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('Specificity:', specificity)
    print('F1 Score:', f1_score)
    return None;

def predict(w, x_training):
    '''
        Function that given the final calculated parameters of w
        and a set of data predicts its estimated outcome.
        Since this prediction involves a logistic regression
        the predicted data array has to be modified, for each
        value less than 0.5 that value is changed to 0 and for
        each value great than or equal to 0.5 that value 
        changes to 1.

        INPUT:
            x_training: numpy array with the scaled values of the training data
            w: numpy array with the values of the parameters

        OUTPUT:
            return the predicted data to each data row
    '''
    ones = np.atleast_2d(np.ones(len(x_training))).T
    x = np.hstack((ones, x_training))

    predicted_data = np.dot(x,w)
    print(predicted_data.T)
    #print(predicted_data)
    predicted_data = np.where(predicted_data < 0.5, 0, 1)

    return predicted_data