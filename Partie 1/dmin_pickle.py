import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time 
import preprocessing as pre
import pickle


print("dmin.py: starting program")

# ---------------------------------
# Loading the test dataset saved by pickle. 
# The train dataset is already processed and will be loaded later 
# time taken: approx. 2s
def loadTestDataFromPickle(test_filename):
	print("-loadTestPickle: loading test data")
	test_data = pickle.load(open(test_filename, "rb"))
	return test_data


# ---------------------------------
# Applying pre-processing to the data and saving it into a pickle file
# We are only saving the test data, as the train data is going to be saved into
# classes (much lighter pickle files)
def applyPreprocessing(test_filename, filter1, filter2, filter3, imageLimit):

	test_data = loadmat('test_32x32.mat')
	pickle.dump(pre.imageProcessingThreeFilters(test_data, filter1, filter2, filter3, imageLimit, 100), open(test_filename, "wb"))
	
	train_data = loadmat('train_32x32.mat')
	pre.imageProcessingThreeFilters(train_data, filter1, filter2, filter3, imageLimit, 100)

	return train_data



# ---------------------------------
# learning function : creates the classes from the train data
# saves the classes into a pck file
def saveAverageLearningVector(train_data, train_filename, imageLimit):
	print("\n---averageLearningVector: creating the model")

	avgVector = {}
	allClassVectors = [[] for i in range(10)] #create 10 lists for the 10 classes

	# putting the images into their own class depending on their label
	# time taken: approx. 0.35s 
	for i in range(imageLimit):
		allClassVectors[train_data['y'][i]-1].append(train_data['X'][:, :, :, i])

	# computing the average of the vectors
	# time taken: approx. 41s. This part takes the longest
	for i in range(10):
		if len(allClassVectors[i]) != 0:
			avgVector[i+1] = np.average(allClassVectors[i], axis=0)

	# saving the classes
	pickle.dump(avgVector, open(train_filename, "wb"))


# ---------------------------------
# learning function : loads the model (see dmin.py for more information)
def loadAverageLearningVector(train_filename):
	print("\n---averageLearningVector: loading the model's classes from pickle")
	return pickle.load(open(train_filename, "rb"))



# ---------------------------------
# compare a picture with the existing model
def findLabel(picture, averageLearningVector):

	label = 1
	frobeniusNorm = np.linalg.norm(picture - averageLearningVector[label])
	for i in range(2, 11):
		if np.linalg.norm(picture - averageLearningVector[i]) < frobeniusNorm:
			frobeniusNorm = np.linalg.norm(picture - averageLearningVector[i])
			label = i
	return label



# ---------------------------------
# main classifier function
def minimumDistanceClassifier(avgVector, test, imageLimit):

	print("\n-minimumDistanceClassifier: start")

	success = 0
	print("--findLabel: finding the distance between the data and the model")

	for i in range(imageLimit):
		label = findLabel(test["X"][:, :, :, i], avgVector)
		if label == test["y"][i]:
			success += 1

	print("--findLabel: end")
	print("-\nminimumDistanceClassifier: end")
	return success



# ---------------------------------
# main
if __name__ == "__main__":
	
	
	print("dmin.py: Starting to compute learning vector")
	
	start = time.time()

	# **********
	# program settings
	imageLimit = 400 # can be = len(train_data['y']) to process ALL the data

	computePreprocessing = True

	train_filename = "train_full_new_hsb.pck"
	test_filename = "test_full_new_hsb.pck"

	filter1 = pre.histogramEqualization
	filter2 = pre.sobelFilter
	filter3 = pre.brightness

	# **********



	# **********
	# this part creates the models and loads the necessary files
	if(computePreprocessing):
		train_data = applyPreprocessing(test_filename, filter1, filter2, filter3, imageLimit)
		saveAverageLearningVector(train_data, train_filename, imageLimit)

	test_data = loadTestDataFromPickle(test_filename) 
	avgVector = loadAverageLearningVector(train_filename)

	# **********




	# **********
	# classify the test data

	if len(test_data["y"]) > imageLimit:
		success = minimumDistanceClassifier(avgVector, test_data, imageLimit)
		successPercentage =  100.*success/imageLimit

		#display the success rate
		print("\ndmin.py: Success rate : " + str(success) + " / " 
			+ str(imageLimit) + 
			" (" + str(successPercentage) + "%)")
	else:
		success = minimumDistanceClassifier(avgVector, test_data, len(test_data["y"]))
		successPercentage =  100.*success/len(test_data["y"]) 

		#display the success rate
		print("\ndmin.py: Success rate : " + str(success) + " / " 
			+ str(len(test_data["y"])) + 
			" (" + str(successPercentage) + "%)")

	end = time.time()

	total = end - start
	print("\ndmin.py: Time taken: " + str(total) + " sec.\n\n")
	# **********





