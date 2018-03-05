"""

#####################

This script defines the class for a naive supervised learning classifier.
The classifier alwasy predicts the class with most values in the training set.

#####################

"""


import pandas as pd
import numpy as np

class NaiveModel():
    
    def __init__(self):
        
        # Initialise most_frequent_class value
        self.most_frequenct_class = None
    
    def fit(self, y_train):
        
        """
        
            Fit the naive benchmark model. Since this model always return the values for the most
            frequent class, there is actuallly no need to look at x_train data.
            
            Args:
                - y_train: a numpy array with all the labels' values
                
            Returns:
                N/A: simply fit the model
        
        """
        
        # Transform numpy array into pandas series
        y_train = pd.Series(y_train)
        
        # Find value counts
        value_counts = y_train.value_counts()
        
        # Update most frequent class
        self.most_frequenct_class = value_counts.idxmax()
        
    
    def predict(self, x_test):
        
        """
        
            Predict the value for set of samples. It simply returns an array with the same number of samples as
            the input array, and whose values are those of the most frequent class identified during fit.
        
            Args:
                - x_test: a numpy array
                
            Returns:
                - predictions: a numpy array
        
        """
        
        if self.most_frequenct_class == None:
            
            print("Fit the model first!")
        
        else:
            
            predictions = np.ones(x_test.shape[0])
            
            predictions = predictions * self.most_frequenct_class
            
            return predictions