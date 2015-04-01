#!/usr/bin/env python 
#!-*- coding: utf-8 -*-

import numpy as np

__author__ = "Irshad Ahmad Bhat, Vaishali Pal"
__version__ = "1.0"
__email__ = ["irshad.bhat@research.iiit.ac.in", "vaishali.pal@research.iiit.ac.in"]

class Report():

    '''classification report in terms of percentage-accuracy and confusion-matrix'''

    def __init__(self, X, Y):

        self.X = np.asarray(X, str)         # true labels
        self.Y = np.asarray(Y, str)         # predicted labels
	self.cnf_matrix = None		    # confusion matrix
        XY = np.unique(np.hstack((X,Y)))
        self.classes_ = np.sort(XY).astype(str)

    def accuracy(self):
        """Claculate percentage-accuracy of the classifier"""
        correct = 0.0
        for true_, predicted in zip(self.X, self.Y):
            if true_ == predicted:
                correct += 1

        return (correct / len(self.X))*100

    def confusion_matrix(self):
        """Claculate confusion-matrix of the classifier"""
        true_count = {k:0 for k in self.classes_}
        confusion_matrix = np.zeros((len(self.classes_),)*2)
        label_id = {k:i for i,k in enumerate(self.classes_)}

        for true_, predicted in zip(self.X, self.Y):
            true_count[true_] += 1
            confusion_matrix[label_id[true_]][label_id[predicted]] += 1

        for k,v in true_count.items():
            confusion_matrix[label_id[k]] /= v
            confusion_matrix[label_id[k]] *= 100

        # format results of confusion-matrix to be more readable
        ordered_labels = sorted(label_id.items(), key=lambda x:x[1])
	width = max(max(len(i) for i in label_id) + 4, 7)
        cnf_out = '\nConfusion Matrix\n\n'+''.ljust(width)
        cnf_out += ''.join([k.ljust(width) for k,v in ordered_labels]) + '\n\n'
        cnf_out += '\n'.join([k.ljust(width)+''.join([str(round(i)).ljust(width) \
                            for i in confusion_matrix[v]]) \
                            for k,v in ordered_labels]) + '\n'

        self.cnf_matrix = np.array([map(round,confusion_matrix[v]) for k,v in ordered_labels])
        return cnf_out

    def precision_recall_fscore(self):
	
	if not self.cnf_matrix:
	    zzz = self.confusion_matrix()

	dim = len(self.classes_)
	precision = np.zeros(dim)
	recall = np.zeros(dim)
	f1_score = np.zeros(dim)

	for i in range(dim):
	    precision[i] = self.cnf_matrix[i][i] / np.sum(self.cnf_matrix[:,i])
	    recall[i] = self.cnf_matrix[i][i] / np.sum(self.cnf_matrix[i])
	    f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

	prf = np.array([precision, recall, f1_score]).T
	prf_in = ['precision', 'recall', 'f1_score']
   
	in_width = max(max(len(i) for i in prf_in) + 4, 7)
	l_width = max(max(len(i) for i in self.classes_) + 4, 11)
        prf_out = '\nMetrics\n\n' + ''.rjust(l_width)
        prf_out += ''.join([k.rjust(in_width) for k in prf_in]) + '\n\n'
        prf_out += '\n'.join([self.classes_[i].rjust(l_width) + \
			''.join([str(round(v,2)).rjust(in_width) for v in prf[i]]) \
                        for i in range(dim)]) + '\n\n'	
	prf_out += 'avg / total'.rjust(l_width) + ''.join([str(round(v,2)).rjust(in_width) \
						for v in np.sum(prf, axis=0) / dim])
 
	return prf_out
