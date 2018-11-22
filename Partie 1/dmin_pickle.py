import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time 
import preprocessing as pre
import pickle

# ---------------------------------
# Loading the test dataset saved by pickle. 
# The train dataset is already processed and will be loaded later 
# time taken: approx. 2s
print("dmin.py: starting program\ndmin.py: loading train data")

# Loading the data (saved by pickle)
test_data = pickle.load(open("pre_test_mat.pck", "rb"))

# Applying pre-processing to the data and saving it into a pickle file
# pre.imageProcessingTwoFilters(train_data, pre.brightness, pre.contrast, 80)
# pickle.dump(pre.imageProcessingTwoFilters(test_data, pre.brightness, pre.contrast, 80), open("pre_test_mat.pck", "wb"))



# ---------------------------------
#learning function : loads the model (see dmin.py for more information)
def averageLearningVector():
	print("\n---averageLearningVector: loading the model from pickle")
	return pickle.load(open("pre_train_class.pck", "rb"))



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
#main classifier function
def minimumDistanceClassifier(test):

	print("\n-minimumDistanceClassifier: start")

	success = 0
	avgVector = averageLearningVector()
	print("--findLabel: finding the distance between the data and the model")

	for i in range(len(test["y"])):
		label = findLabel(test["X"][:, :, :, i], avgVector)
		if label == test["y"][i]:
			success += 1

	print("--findLabel: end")
	print("-minimumDistanceClassifier: end")
	return success


# ---------------------------------
# main
if __name__ == "__main__":
	
	
	print("dmin.py: Starting to compute learning vector")
	
	start = time.time()

	success = minimumDistanceClassifier(test_data)
	successPercentage =  100.*success/len(test_data["y"])

	print("\ndmin.py: Success rate : " + str(success) + " / " 
		+ str(len(test_data["y"])) + 
		" (" + str(successPercentage) + "%)")
	#classes = initializeClasses(train_data)

	end = time.time()

	total = end - start
	print("dmin.py: Time taken: " + str(total) + " sec.")
	
