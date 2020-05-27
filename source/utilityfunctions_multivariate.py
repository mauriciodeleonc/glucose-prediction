'''
    utilityfunctions.py
    Python library with many ML functions
    including linear regression, gradient
    descent, L2 norm, amongst others.

    Authors: 
        * Mauricio A. De León Cárdenas 505597
        * Juan M. Álvarez Sánchez 511385
        * Viviana Vázquez Gómez Martínez 509271
        * Orlando X. Torres Guerra 513341
    Institution: Universidad de Monterrey
    First created: Thu 14 May 2020
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
        w params array can be multiplied

        INPUT:
            x: numpy array with the scaled values of x
            w: numpy array with the values of the parameters

        OUTPUT:
            return the multiplication of both arrays
    '''
    return np.matmul(x.T, w)

def compute_gradient_of_cost_function_multivariate(x, y, w):
    '''
        Function that computes the gradient of the cost function

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
    gradient_of_cost_function = (np.matmul(residual.T, x.T) / N)

    return gradient_of_cost_function

def compute_L2_norm_multivariate(gradient_of_cost_function, x):
    '''
        Function that calulates the L2 Norm value,
        which will be compared to the stopping criteria 
        allowing the gradient descent to decide when to stop.

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
    L2_norm = 100.0

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

def predict(w, x_training):
    '''
        Function that given the final calculated parameters of w
        and a set of data predicts its estimated value

        INPUT:
            x_training: numpy array with the scaled values of the training data
            w: numpy array with the values of the parameters

        OUTPUT:
            return the predicted insulin for each data row
    '''
    ones = np.atleast_2d(np.ones(len(x_training))).T
    x = np.hstack((ones, x_training))

    prices = np.matmul(x,w)

    return prices