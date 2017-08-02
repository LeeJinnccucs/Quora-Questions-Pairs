import numpy as np
import pandas as pd
import math
from preProcessing import preProcessing
from fuzzywuzzy import fuzz

class fuzzy_features:

	def __init__(self, inputData):
		self.questions = inputData.questions
		self.crash = inputData.crash_questions
		#features list
		self.qratio = list()
		self.wratio = list()
		self.partialRatio = list()
		self.partialTokenSet = list()
		self.partialTokenSort = list()
		self.tokenSet = list()
		self.tokenSort = list()
		self.allfuzzy = list()
		self.fuzzRatio()
		self.check_nan()
	
	def fuzzRatio(self): #calculate features
		for q in self.questions:
			self.qratio.append(fuzz.QRatio(''.join(q[0]), ''.join(q[1])))
			self.wratio.append(fuzz.WRatio(''.join(q[0]), ''.join(q[1])))
			self.partialRatio.append(fuzz.partial_ratio(''.join(q[0]), ''.join(q[1])))
			self.partialTokenSet.append(fuzz.partial_token_set_ratio(''.join(q[0]), ''.join(q[1])))
			self.partialTokenSort.append(fuzz.partial_token_sort_ratio(''.join(q[0]), ''.join(q[1])))
			self.tokenSet.append(fuzz.token_set_ratio(''.join(q[0]), ''.join(q[1])))
			self.tokenSort.append(fuzz.token_sort_ratio(''.join(q[0]), ''.join(q[1])))

		for index in self.crash: #fill crash
			self.qratio.insert(index, -1)
			self.wratio.insert(index, -1)
			self.partialRatio.insert(index, -1)
			self.partialTokenSet.insert(index, -1)
			self.partialTokenSort.insert(index, -1)
			self.tokenSet.insert(index, -1)
			self.tokenSort.insert(index, -1)
		
		self.allfuzzy.append(self.qratio)
		self.allfuzzy.append(self.wratio)
		self.allfuzzy.append(self.partialRatio)
		self.allfuzzy.append(self.partialTokenSet)
		self.allfuzzy.append(self.partialTokenSort)
		self.allfuzzy.append(self.tokenSet)
		self.allfuzzy.append(self.tokenSort)
	
	def check_nan(self): #check nan
		nan = float('nan')
		for a in self.allfuzzy:
			if nan in a:
				print ('oh no')

def main():
	data = pd.read_csv('../data/train.csv').values[:, 3:]
	a = preProcessing(data)
	bb = fuzzy_features(a)
	print bb.qratio

if __name__ == "__main__":
	main()
