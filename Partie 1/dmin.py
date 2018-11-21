import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time 
import preprocessing as pre


# Loading the datasets
print("--dmin.py: starting program")
print("--dmin.py: loading train data")

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

# Applying pre-processing to the data
#pre.imageProcessing(train_data, pre.contrast)
#pre.imageProcessing(test_data, pre.contrast)

#learning function : appends every images into separate lists
#and computes the average of the class
def averageLearningVector(data):

	print("\n--averageLearningVector: start")

	avgVector = {}
	allClassVectors = [[] for i in range(10)] #create 10 lists for the 10 classes


	#putting the images into their own class depending on their label
	#time : approx. 0.35s 
	for i in range(len(data['y'])):
		allClassVectors[data['y'][i]-1].append(data['X'][:, :, :, i])


	#computing the average of the vectors
	for i in range(10):
		if len(allClassVectors[i]) != 0:
			print("Yes!")
			avgVector[i+1] = np.mean(allClassVectors[i], axis=0)



	print("--averageLearningVector: end")

	return avgVector

#
def findLabel(picture, averageLearningVector):
	label = 1
	frobeniusNorm = np.linalg.norm(picture - averageLearningVector[label])
	for i in range(2, 11):
		if np.linalg.norm(picture - averageLearningVector[i]) < frobeniusNorm:
			frobeniusNorm = np.linalg.norm(picture - averageLearningVector[i])
			label = i

	return label

#
def minimumDistanceClassifier(test, train):

	print("\n--minimumDistanceClassifier: start")

	success = 0
	avgVector = averageLearningVector(train)

	for i in range(len(test["y"])):
		label = findLabel(test["X"][:, :, :, i], avgVector)
		if label == test["y"][i]:
			success += 1

	print("--minimumDistanceClassifier: end")

	return success

if __name__ == "__main__":
	
	
	print("--dmin.py: Starting to compute learning vector")
	
	start = time.time()

	success = minimumDistanceClassifier(test_data, train_data)
	successPercentage =  100.*success/len(test_data["y"])

	print("\n--dmin.py: Success rate : " + str(success) + " / " 
		+ str(len(test_data["y"])) + 
		" (" + str(successPercentage) + "%)")
	#classes = initializeClasses(train_data)

	end = time.time()

	total = end - start
	print("--dmin.py: Time taken: " + str(total) + " sec.")
	
