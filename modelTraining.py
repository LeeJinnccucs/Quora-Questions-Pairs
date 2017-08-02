import numpy as np
import pandas as pd
import math
from sklearn.naive_bayes import GaussianNB

def modelTraining(inputVector, sim, test):
	gnb = GaussianNB()
	gnb.fit(inputVector, sim)
	result = gnb.predict(test)
	return result


	
