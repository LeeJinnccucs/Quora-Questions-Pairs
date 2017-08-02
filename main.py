import numpy as np
import pandas as pd
import math
import os
from preProcessing import preProcessing
from relativeClause import relative_processing
from vsm_similarity import vsm_similarity
from modelTraining import modelTraining
from knnTraining import knnTraining
from gensimModel import gensimModel
from fuzzywuzzy_features import fuzzy_features
from xgbTraining import xgbTraining

def main():
	train = pd.read_csv('../data/train.csv').values[:, 3:] #read input
	print('start preprocessing')
	train_data = preProcessing(train) #preprocessing
#	print len(train_data.questions)
	print ('start similarity computing')
	train_similarity = vsm_similarity(train_data) #similarity features
#	print len(train_similarity.cosineValue)
	sim = train_data.outcome #outcome
#	print len(sim)
	print ('start gensim Model')
	train_gensim = gensimModel(train_data) #word2vec features
	print ('start fuzzy')
	train_fuzzy = fuzzy_features(train_data) #fuzzy features
#	print train_data.allBasic
	train_data.allBasic.extend(train_similarity.allSimilarity)  #combine features
#	print train_data.allBasic
	train_data.allBasic.extend(train_gensim.allWord2Vec) 
#	print train_data.allBasic
	train_data.allBasic.extend(train_fuzzy.allfuzzy)
#	print train_data.allBasic
	train_in = zip(*train_data.allBasic)    #zip into whole list
	train_input = np.array(train_in)
	print ('train input:')
#	print train_input
#	print len(train_input)
	test = pd.read_csv('../data/test.csv').values[:, 1:] #read test data
	print('start preprocessing')
	test_data = preProcessing(test)  #preprocessing
#	print len(test_data.questions)
	print('start similarity computing')
	test_similarity = vsm_similarity(test_data)	#similarity features
#	print len(test_similarity.cosineValue)
	print ('start gensim Model')
	test_gensim = gensimModel(test_data) #word2vec features
	print ('start fuzzy')
	test_fuzzy = fuzzy_features(test_data) #fuzzy features
	test_data.allBasic.extend(test_similarity.allSimilarity) #combine features
	test_data.allBasic.extend(test_gensim.allWord2Vec)
	test_data.allBasic.extend(test_fuzzy.allfuzzy)
	test_in = zip(*test_data.allBasic)  #zip into list
	test_input = np.array(test_in)
	print ('test_input:')
#	print test_input
#	print len(test_input)
	result = xgbTraining(train_input, sim, test_input) #feed into model
#	print result

	#make outout file
	outputArray = []
	Index = []
	for ind, x in enumerate(result):
		outputArray.append([])
		outputArray[ind].append(ind)
		Index.append(ind+1)
		outputArray[ind].append(x)
	output = np.asarray(outputArray)
	columns = ['test_id', 'is_duplicate']
	outputDf = pd.DataFrame(output, columns = columns, index = Index)
	outputDf.to_csv('output.csv', index = False)

if __name__ == "__main__":
	main()
