import numpy as np
import pandas as pd
import math
import xgboost as xgb

def xgbTraining(inputVector, sim, test):
	model = xgb.XGBClassifier(max_depth = 5, eta = 0.03, objective = 'binary:logistic', eval_metric = 'logloss')
	model.fit(inputVector, sim)
	result = model.predict(test)
	return result
