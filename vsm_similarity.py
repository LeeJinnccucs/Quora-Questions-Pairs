import numpy as np
import pandas as pd
import math
import util
import scipy as sp
from preProcessing import preProcessing
from sklearn.metrics import jaccard_similarity_score

class vsm_similarity:
	
	def __init__(self, inputData):
#		a = preProcessing()
		self.questions = None
		self.cosineValue = list() #cosine similarity
		self.crash = None #crash index
		self.cityblockValue = list() #cityblock similarity
		self.jaccardValue = list() #jaccard
		self.canberraValue = list() #canberra
		self.euclideanValue = list() #euclidean
		self.count = 0
		print ("finding cosine vector")
		self.questions = inputData.questions
		self.crash = inputData.crash_questions #crash index
		self.allSimilarity = list()
		self.makeVector()

	def makeVector(self): #make features vector
		for q in self.questions:
			#make vsm model
			base = q[0] + q[1]
			KeywordIndex = set((item for item in base))
			vectorIndex = {}
			offset = 0
			for word in KeywordIndex:
				vectorIndex[word] = offset
				offset += 1
			vectorA = [0]*len(KeywordIndex)
			vectorB = [0]*len(KeywordIndex)
			for word in q[0]:
				vectorA[vectorIndex[word]] += 1
			for word in q[1]:
				vectorB[vectorIndex[word]] += 1
			self.cityblockValue.append(sp.spatial.distance.cityblock(vectorA,vectorB))
			self.jaccardValue.append(sp.spatial.distance.jaccard(vectorA,vectorB))
			self.canberraValue.append(sp.spatial.distance.canberra(vectorA,vectorB))
			self.euclideanValue.append(sp.spatial.distance.euclidean(vectorA,vectorB))
			if sum(vectorA) == 0 and sum(vectorB) == 0:
				self.cosineValue.append(0)
			elif (sum(vectorA)*sum(vectorB)) == 0 and (sum(vectorA)+sum(vectorB)) != 0:
				self.cosineValue.append(0)
			else:
				self.cosineValue.append(util.cosine(vectorA, vectorB))
		for index in self.crash: #fill the crash part
			self.cosineValue.insert(index, -1)
			self.cityblockValue.insert(index, -1)
			self.jaccardValue.insert(index, -1)
			self.canberraValue.insert(index, -1)
			self.euclideanValue.insert(index, -1)
		
		self.allSimilarity.append(self.cosineValue)
		self.allSimilarity.append(self.cityblockValue)
		self.allSimilarity.append(self.jaccardValue)
		self.allSimilarity.append(self.canberraValue)
		self.allSimilarity.append(self.euclideanValue)

def main():
	data = pd.read_csv('../data/train.csv').values[:, 3:]
	a = preProcessing(data)
	qq = cosineVector(a)
#	print qq.cosineValue
#	print len(qq.cosineValue)

if __name__ == "__main__":
	main()
