'''
    logistic-classification.py
    Python script for classifying data 
    using logistic classification.

    Author: Mauricio A. De Leon Cardenas
    Email: mauricio.deleonc@udem.edu
    Institution: Universidad de Monterrey
    First created: Sun 12 Mar 2020
'''

#import libraries
import numpy as np
import utilityfunctions as uf
import time
from tabulate import tabulate

#Variable that stores start time of execution
start = time.time()

def main():
    '''
        Driver code for the logistic regression algorithm.
    '''

    #Load data 
    x_data, y_data, mean, std = uf.load_data_multivariate('diabetes.csv')

    #Split the total data into training and testing data arrays using 80% for training and 20% for testing
    x_training, y_training, x_testing, y_testing = uf.split_data(x_data, y_data, 0.95)

    #Scale both training and testing data
    x_training_scaled = uf.scale_data(x_training, mean, std)
    x_testing_scaled = uf.scale_data(x_testing, mean, std)
    
    #Define stopping criteria and learning rate
    stopping_criteria = 0.01
    learning_rate = 0.0005

    #Define array of w params
    w_params = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])

    #Fill w params array
    w_params, iterations = uf.gradient_descent_multivariate(x_training_scaled, y_training, w_params, stopping_criteria, learning_rate)

    #Predict outcome based on the values of w and the x testing data
    predicted_data = uf.predict(w_params, x_testing_scaled)
    
    #Fill confusion matrix with the actuall data and the predicted data
    confusion_matrix = uf.get_confusion_matrix(y_testing, predicted_data)
    
    printing_flag = True

    #Print all generated data
    if(printing_flag):
        #Print w parameters found
        i = 0
        for w in w_params:
            print('w' + str(i) + ': ', str(w))
            i = i + 1

        #Print confusion matrix. Columns are actual class and rows are predicted class
        print('\n' + 80*'-' + '\n' + 'Confusion matrix\n' + 80*'-')
        l = [['Has diabetes (1):', confusion_matrix[0], confusion_matrix[1]], [ "Doesn't have diabetes (0):", confusion_matrix[2], confusion_matrix[3]]]
        table = tabulate(l, headers=['Has diabetes (1):', "Doesn't have diabetes (0):"], tablefmt='orgtbl')
        print(table, '\n')

        uf.print_performance_metrics(confusion_matrix)
    

    #Print iterations generated
    print('\n' + str(iterations), 'iterations')

    #Print total time of execution
    print(time.time() - start,'seconds')
    return None

#call main function
main()
