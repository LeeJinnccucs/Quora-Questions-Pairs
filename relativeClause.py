import numpy as np
import pandas as pd
import math
import util
from preProcessing import preProcessing

class relative_processing:

	crashed = None
	question_data = None
	relative_keyword = ['how', 'where', 'who', 'when', 'why', 'what']
	relative_cosineVector = []

	def __init__(self, inputData):
		self.crashed = None
		self.question_data = None
		self.relative_cosineVector = list()
		print ("finding relative vector")
		self.question_data = inputData.questions
		self.crashed = inputData.crash_questions
		self.findRelative()

	def findRelative(self):
		for q in self.question_data:
			vectorQ1 = [0]*len(self.relative_keyword)
			vectorQ2 = [0]*len(self.relative_keyword)
			for index, word in enumerate(self.relative_keyword):
				if word in q[0]:
					vectorQ1[index] += 1
				if word in q[1]:
					vectorQ2[index] += 1
			if sum(vectorQ1) == 0 and sum(vectorQ2) == 0:
				self.relative_cosineVector.append(0)
			elif (sum(vectorQ1)*sum(vectorQ2))==0 and (sum(vectorQ1)+sum(vectorQ2))!=0:
				self.relative_cosineVector.append(0)
			else:
				gj = util.cosine(vectorQ1, vectorQ2)
#				if gj > 0.99:
#					self.relative_cosineVector.append(1)
#				else:
#					self.relative_cosineVector.append(0)
				self.relative_cosineVector.append(util.cosine(vectorQ1, vectorQ2))
		for index in self.crashed:
			self.relative_cosineVector.insert(index, 0.0)	

def main():
	data = pd.read_csv('../data/train.csv').values[:, 3:]
	a = preProcessing(data)
	gg = relative_processing(a)
	print gg.relative_cosineVector

if __name__ == "__main__":
	main()
