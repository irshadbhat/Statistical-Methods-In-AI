#!/usr/bin/env python 
#!-*- coding: utf-8 -*-

import numpy as np

__author__ = "Irshad Ahmad Bhat"
__version__ = "1.0"
__email__ = "irshad.bhat@research.iiit.ac.in"


class Report():

    '''classification report in terms of percentage-accuracy and confusion-matrix'''

    def __init__(self, X, Y):
	
	self.X = np.asarray(X, str)	    # true labels
	self.Y = np.asarray(Y, str)	    # predicted labels
	self.classes_ = set(self.X) | set(self.Y)
    
    def accuracy(self):

	"""Claculate percentage-accuracy of the classifier"""	

	correct = 0.0
	for true_, predicted in zip(self.X, self.Y):
	    if true_ == predicted:
		correct += 1

	print '\naccuracy = {}\n'.format((correct / len(self.X))*100)

    def confusion_matrix(self):

	"""Claculate confusion-matrix of the classifier"""

	true_count = {k:0 for k in self.classes_}
	cf_matrix = np.zeros((len(self.classes_),)*2)
	label_id = {k:i for i,k in enumerate(self.classes_)}
    
	for true_, predicted in zip(self.X, self.Y):
	    true_count[true_] += 1
	    cf_matrix[label_id[true_]][label_id[predicted]] += 1
	
	for k,v in true_count.items():
	    cf_matrix[label_id[k]] /= v
	    cf_matrix[label_id[k]] *= 100
    
	# format results of confusion-matrix to be more readable 
	label_id = {k:i for i,k in enumerate(self.classes_)}
	ordered_labels = sorted(label_id.items(), key=lambda x:x[1])
	width = max(max(len(i) for i in label_id) + 4, 7)
	cf_report = '\n'+''.ljust(width)
	cf_report += ''.join([k.ljust(width) for k,v in ordered_labels]) + '\n\n'
	cf_report += '\n'.join([k.ljust(width)+''.join([str(round(i)).ljust(width) \
			    for i in cf_matrix[v]]) \
			    for k,v in ordered_labels]) + '\n'

	print '\nConfusion Matrix'
	print cf_report

