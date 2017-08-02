import gensim
import logging
import numpy as np
import pandas as pd
import math
import scipy as sp
from preProcessing import preProcessing
from sklearn.metrics import jaccard_similarity_score



#logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

class gensimModel:
	
	def __init__(self, inputData):
		print ("loading google word2vec model")
		self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)
		self.questions = inputData.questions
		self.crash = inputData.crash_questions
		self.word2vecList = list()
		self.cosineValue = list() #cosine similarity
		self.cityblockValue = list() #cityblock similarity
		self.jaccardValue = list() #jaccard
		self.canberraValue = list() #canberra
		self.euclideanValue = list() #euclidean
		self.allVec = list() #temp with nan
		self.makeQuestionVec()
		self.cal_similarity()
		self.allWord2Vec = list() #list of list of features
		self.check_nan()

	def avg_feature_vec(self, words, model, num_features): #calculate sentence vector
		featureVec = np.zeros((num_features,), dtype = 'float32')
		num = 0

		for word in words:
			if word in model:
				num = num+1
				featureVec = np.add(featureVec, model[word])

		if (num > 0):
			featureVec = np.divide(featureVec, num)
		return featureVec
	
	def makeQuestionVec(self): #turn all quesitons into word2vec vector
		self.word2vecList = [[self.avg_feature_vec(q[0], self.model, 300), self.avg_feature_vec(q[1], self.model, 300)] for q in self.questions]
		
	def cal_similarity(self): #similarity calculating
		for vec in self.word2vecList:
			self.cosineValue.append(sp.spatial.distance.cosine(vec[0], vec[1]))
			self.cityblockValue.append(sp.spatial.distance.cityblock(vec[0], vec[1]))
			self.jaccardValue.append(sp.spatial.distance.jaccard(vec[0], vec[1]))
			self.canberraValue.append(sp.spatial.distance.canberra(vec[0], vec[1]))
			self.euclideanValue.append(sp.spatial.distance.euclidean(vec[0], vec[1]))
		for index in self.crash: #fill crash
			self.cosineValue.insert(index, -1)
			self.cityblockValue.insert(index, -1)
			self.jaccardValue.insert(index, -1)
			self.canberraValue.insert(index, -1)
			self.euclideanValue.insert(index, -1)
		
		self.allVec.append(self.cosineValue)
		self.allVec.append(self.cityblockValue)
		self.allVec.append(self.jaccardValue)
		self.allVec.append(self.canberraValue)
		self.allVec.append(self.euclideanValue)

	def check_nan(self): # fill nan for similarity
		self.allWord2Vec = [[float(-1) if (b == float('nan'))  or (np.isnan(b)) else b for b in a] for a in self.allVec]
"""
	for a in self.allWord2Vec:
			for b in a:
				print type(b)
				print b
				if (b == float('nan')) or (np.isnan(b)):
					print ('gg')
"""
#				if not isinstance(b, float):
#					print ('no float')

def main():
	
	data = pd.read_csv('../data/train.csv')
	raw_question = data.values[:, 3:]
	train_data = preProcessing(raw_question)
	bb = gensimModel(train_data)
	
#	print model.similarity('woman', 'man')
#	print bb.allWord2Vec

if __name__ == "__main__":
	main()
