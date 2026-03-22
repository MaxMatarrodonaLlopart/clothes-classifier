__authors__ = ['1669698', '1668784']
__group__ = '61'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        if not np.issubdtype(train_data.dtype, np.floating):
            train_data = train_data.astype(float)
        
        shape = train_data.shape
        if len(shape) > 2:
            self.train_data = np.reshape(train_data, (shape[0],shape[1]*shape[2]))

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        shape = test_data.shape
        if len(shape) > 2:
            data = np.reshape(test_data, (shape[0],shape[1]*shape[2]))
        
        distances = cdist(data, self.train_data)
        
        self.neighbours = np.argsort(distances, axis=1)[:, :k]
        self.neighbors = np.array([self.labels[i] for i in self.neighbours])


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        sorted_classes = np.zeros((self.neighbors.shape[0],), dtype='<U10')
        
        for i in range(self.neighbors.shape[0]):
            dict_count = {}
            for class_type in self.neighbors[i]:
                if class_type not in dict_count:
                    dict_count[class_type] = 0
                dict_count[class_type] += 1
            
            sorted_classes[i] = max(dict_count.items(), key=operator.itemgetter(1))[0]

            

            
        return sorted_classes

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
