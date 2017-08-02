import numpy as np
import pandas as pd
import math
from sklearn.neighbors import KNeighborsClassifier

def knnTraining(inputVector, sim, test):
	neigh = KNeighborsClassifier(n_neighbors = 4)
	neigh.fit(inputVector, sim)
	result = neigh.predict(test)
	return result
