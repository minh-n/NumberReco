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
pre.imageProcessing(train_data, pre.histogramEqualization)
pre.imageProcessing(test_data, pre.histogramEqualization)


def averageLearningVector(data):

	print("--averageLearningVector: start")

	avgVector = {}

	allVectorClass1 = []
	allVectorClass2 = []
	allVectorClass3 = []
	allVectorClass4 = []
	allVectorClass5 = []
	allVectorClass6 = []
	allVectorClass7 = []
	allVectorClass8 = []
	allVectorClass9 = []
	allVectorClass10 = []

	allVector = []
	allVector.append(allVectorClass1)
	allVector.append(allVectorClass2)
	allVector.append(allVectorClass3)
	allVector.append(allVectorClass4)
	allVector.append(allVectorClass5)
	allVector.append(allVectorClass6)
	allVector.append(allVectorClass7)
	allVector.append(allVectorClass8)
	allVector.append(allVectorClass9)
	allVector.append(allVectorClass10)

	for i in range(len(data['y'])):
		for j in range(1, 11):
			if data['y'][i] == j:
				allVector[j-1].append(data['X'][:, :, :, i])

	for i in range(10):
		if len(allVector[i]) != 0:
			avgVector[i+1] = np.average(allVector[i], axis=0)

	print("--averageLearningVector: end")

	return avgVector

def findLabel(picture, averageLearningVector):
	label = 1
	frobeniusNorm = np.linalg.norm(picture - averageLearningVector[label])
	for i in range(2, 11):
		if np.linalg.norm(picture - averageLearningVector[i]) < frobeniusNorm:
			frobeniusNorm = np.linalg.norm(picture - averageLearningVector[i])
			label = i

	return label

def minimumDistanceClassifier(test, train):
	success = 0
	avgVector = averageLearningVector(train)

	for i in range(len(test["y"])):
		label = findLabel(test["X"][:, :, :, i], avgVector)
		if label == test["y"][i]:
			success += 1

	return success

if __name__ == "__main__":
	
	start = time.time()

	print("--dmin.py: Starting to compute learning vector")
	
	success = minimumDistanceClassifier(test_data, train_data)
	successPercentage =  100*success/len(test_data["y"])

	print("\n--dmin.py: Success rate : " + str(success) + " / " + str(len(test_data["y"])) + " (" + str(successPercentage) + "%)")
	#print("Success rate : %d / %d  (%0.2f \%)" % (success, len(test_data["y"]), success * 100/len(test_data["y"])))
	#classes = initializeClasses(train_data)

	end = time.time()

	end = end - start
	end = end/60
	print("--dmin.py: Time taken: ")
	print(end)
	
