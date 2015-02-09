#!/usr/bin/env python 
#!-*- coding: utf-8 -*-

import numpy as np

__author__ = "Irshad Ahmad Bhat"
__version__ = "1.0"
__email__ = "irshad.bhat@research.iiit.ac.in"

class Distance():

    def euclidean(self, X, Y):

	"""Returns Euclidean distance between to n-dimentional vectors."""

	X = np.asarray(X)	
	Y = np.asarray(Y)	

	return np.sqrt(np.sum((X-Y)**2))

    def manhattan(self, X, Y):

	"""Returns Euclidean distance between to n-dimentional vectors."""

	X = np.asarray(X)	
	Y = np.asarray(Y)
	
	return np.sum(np.abs(X-Y))

    def cosine(self, X, Y):

	"""Returns Cosine distance between to n-dimentional vectors."""

	X = np.asarray(X)	
	Y = np.asarray(Y)	

	return 1 - np.sum(X*Y) / (np.sqrt(np.sum(X**2)) * np.sqrt(np.sum(Y**2)))

