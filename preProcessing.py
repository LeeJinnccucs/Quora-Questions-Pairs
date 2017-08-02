import numpy as np
import pandas as pd
from Parser import Parser
import math
import util

class preProcessing:



	def __init__ (self, rawQuestions):
		self.questions = list() #question after stemmed
		self.idf_documents = ""
		self.count = 0
		self.raw_question = None #q1, q1 and isduplicate
		self.crash_questions = [] #find weird question
		self.isTrain = False  #train or test data
		self.parser = Parser() #from parser.py
		self.outcome = list() #is_duplicated
		self.len_word_q1 = list() #word length feature
		self.len_word_q2 = list() #word length feature
		self.common_words = list() #common words num
		self.result = list()
		self.allBasic = list()   #basic features list
		print('Load training data')
		if len(rawQuestions[0]) > 2:  #for recording outcome in train data
			self.isTrain = True
		self.raw_question = rawQuestions #data from file
		self.stemQuestion(self.raw_question) #stemming
		self.allBasic.append(self.len_word_q1)
		self.allBasic.append(self.len_word_q2)
		self.allBasic.append(self.common_words)
	
	def stemQuestion(self, question):
		for index,q in enumerate(question):
			if type(q[0]) == float or type(q[1]) == float:  #find crashed index
				self.crash_questions.append(index)
				if self.isTrain:
					self.outcome.append(q[2])
				self.len_word_q1.append(0)
				self.len_word_q2.append(0)
				self.common_words.append(0)
				continue
			questions_temp = list()
			q[0] = self.parser.tokenise(q[0])
			q[1] = self.parser.tokenise(q[1])
			#find features
			self.len_word_q1.append(len(q[0]))
			self.len_word_q2.append(len(q[1]))
			self.common_words.append(len(list(set(q[0]).intersection(set(q[1])))))
			#stop word
			q[0] = self.parser.removeStopWords(q[0])
			q[1] = self.parser.removeStopWords(q[1])
			#question list
			questions_temp.append(q[0])
			questions_temp.append(q[1])
			self.idf_documents.join(q[0])
			self.idf_documents.join(q[1])
			self.count = self.count+1
			self.questions.append(questions_temp)
			if self.isTrain == True:
				self.outcome.append(q[2])


def main():
	data = pd.read_csv('../data/train.csv')
	raw_question = data.values[:, 3:]
	train_data = preProcessing(raw_question)
#	print len(train_data.questions)
#	print train_data.crash_questions
#	print len(train_data.outcome)
#	print train_data.questions

if __name__ == "__main__":
	main()
