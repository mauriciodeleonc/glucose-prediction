'''
    multivariate-linear-regression.py
    Python script for testing a multivariate
    linear regression using the gradient
    descent algorithm to predict peak blood
    glucose level 2 hours after eating.

    Authors: 
        * Mauricio A. De León Cárdenas 505597
        * Juan M. Álvarez Sánchez 511385
        * Viviana Vázquez Gómez Martínez 509271
        * Orlando X. Torres Guerra 513341
    Institution: Universidad de Monterrey
    First created: Thu 14 May 2020
'''

#import libraries
import numpy as np
import utilityfunctions_multivariate as uf
import time

#Variable that stores start time of execution
start = time.time()

def main():
    '''
        Driver code for the multivariate linear regression.
    '''

    #Load training data 
    x_data, y_data, mean, std = uf.load_data_multivariate('../datasets/glucose.csv')
    
    #Split the total data into training and testing data arrays using 80% for training and 20% for testing
    x_training, y_training, x_testing, y_testing = uf.split_data(x_data, y_data, 0.8)

    #Scale both training and testing data
    x_training_scaled = uf.scale_data(x_training, mean, std)
    x_testing_scaled = uf.scale_data(x_testing, mean, std)

    #Define stopping criteria and learning rate
    stopping_criteria = 0.01
    learning_rate = 0.0005

    #Define array of w params
    w_params = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])

    #Fill w params array
    w_params, iterations = uf.gradient_descent_multivariate(x_training_scaled, y_training, w_params, stopping_criteria, learning_rate)


    #Predict last mile cost from testing data
    predicted_glucose = uf.predict(w_params, x_testing_scaled)

    printing_flag = True

    #Print all generated data
    if(printing_flag):
        print(32*'-' + '\n' + 'Training data and Y outputs\n' + 32*'-')
        for x,y in zip(x_training, y_training):
            print(x,y)

        print('\n' + 32*'-' + '\n' + 'Training data scaled\n' + 32*'-')
        print(x_training_scaled)

        print('\n' + 32*'-' + '\n' + 'W parameters:\n' + 32*'-')
        i = 0
        for w in w_params:
            print('w' + str(i) + ': ', str(w))
            i = i + 1
        
        print('\n' + 32*'-' + '\n' + 'Testing data\n' + 32*'-')
        print(x_testing)

        print('\n' + 32*'-' + '\n' + 'Testing data scaled\n' + 32*'-')
        print(x_testing_scaled)
        
        print('\n' + 32*'-' + '\n' + 'Predicted glucose outputs\n' + 32*'-')
        print(predicted_glucose.T)
        print('\n')

        '''
        print(y_testing.shape)
        print(float(len(y_testing)))
        print(len(y_testing))
        print((y_testing-predicted_glucose))
        print("mape")
        print(np.mean(np.abs((y_testing - predicted_glucose) / y_testing)) * 100)
        '''
    
    #Print iterations generated
    print(iterations, 'iterations')

    #Print total time of execution
    print(time.time() - start,'seconds')
    return None

#call main function
main()
