1. main.py would start the whole project


2. File for preprocessing:

	* english.stop is for stop word list

	* GoogleNews data is for Word2Vec model training

	* too large to push


3. Quora data train.csv and test.csv are too large, so I won’t upload it,
   
   but there would be sample_submission.csv and output.csv as reference for output format. 


4. Parser.py util.py PorterStemmer.py are for stemming


5. Features mining

	* preProcessing.py deals with raw data and find basic features

	* gensimModel.py is for finding Word2Vec features

	* vsm_similarity.py builds a VSM model and calculates similarity of vectors

	* fuzzywuzzy_features.py deals with FuzzyWuzzy features


6. Model training

	* xgbTraining.py knnTraining.py modelTraining.py(naive space) are models for 

	training

7. Ignore .pyc files
