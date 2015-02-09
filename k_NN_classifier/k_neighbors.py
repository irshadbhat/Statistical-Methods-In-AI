#!/usr/bin/env python 
#!-*- coding: utf-8 -*-

"""
K-Nearest Neighbour classifier

The most naive and intuitive classifier is the nearest neighbour classifier.
k-nearest neighbour classifier is the one that assigns a point x to the most 
frequent class of its k closest neighbor in the feature space.
"""

from collections import Counter

from spatial_distance import Distance 

import numpy as np

__author__ = "Irshad Ahmad Bhat"
__version__ = "1.0"
__email__ = "irshad.bhat@research.iiit.ac.in"


class K_NN():

    def __init__(self, K = 3, metric = 'euclidean'):

	self.K = K
	self.metric = metric
	self.training_data = None
	self.training_labels = None

    def fit(self, X, y):

	self.training_data = np.asarray(X)
	self.training_labels = np.asarray(y)

    def get_metric(self):

	"""Set default distance matrix to be used for calculating 
	distances between two vectors."""

        if self.metric == 'euclidean':
            dis_func = Distance().euclidean
        elif self.metric == 'cosine':
            dis_func = Distance().cosine
        elif self.metric == 'manhattan':
            dis_func = Distance().manhattan
	else:
	    print "Please select a valid distance-metric"
	    return
	return dis_func

    def predict(self, testing_data):
	
	"""Returns a list of predicted labels using k-nearest neighbor classifier"""
    
	labels = []
	dis_func = self.get_metric()
	testing_data = np.asarray(testing_data)

	if len(testing_data.shape) == 1 or testing_data.shape[1] == 1:
	    testing_data = testing_data.reshape(1,len(testing_data))

	for i,vec1 in enumerate(testing_data):
	    # initialize K nearest neighbors with large distances
	    neighbor_ids = [0]*self.K
	    min_distance = np.zeros(self.K) + float('inf')   
	    for j,vec2 in enumerate(self.training_data):
		distance = dis_func(vec1,vec2)
		if np.any(distance < min_distance):
		    index = np.argmax(min_distance)
		    min_distance[index] = distance
		    neighbor_ids[index] = j
	    neighbors = self.training_labels[neighbor_ids]
	    # get most common label
	    most_common = Counter(neighbors).most_common()[0][0]
	    labels.append(most_common)

	return labels
