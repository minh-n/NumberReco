import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time 

print("--Classif.py: starting program")
print("--Classif.py: loading train data")

train_data = loadmat('train_32x32.mat')
#test_data = loadmat('test_32x32.mat')

labels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def createLearningVector(picture):
	vector = []

	lines = np.shape(picture)[0]
	columns = np.shape(picture)[0]

	for i in range(lines):
		for j in range(columns):
			for c in range(3):
				vector.append(picture[i, j, c])

	return vector

def averageLearningVector(data):
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

	for i in range(len(data['y'])):
		if data['y'][i] == 1:
			allVectorClass1.append(createLearningVector(data['X'][:, :, :, i]))
		elif data['y'][i] == 2:
			allVectorClass2.append(createLearningVector(data['X'][:, :, :, i]))
		elif data['y'][i] == 3:
			allVectorClass3.append(createLearningVector(data['X'][:, :, :, i]))
		elif data['y'][i] == 4:
			allVectorClass4.append(createLearningVector(data['X'][:, :, :, i]))
		elif data['y'][i] == 5:
			allVectorClass5.append(createLearningVector(data['X'][:, :, :, i]))
		elif data['y'][i] == 6:
			allVectorClass6.append(createLearningVector(data['X'][:, :, :, i]))
		elif data['y'][i] == 7:
			allVectorClass7.append(createLearningVector(data['X'][:, :, :, i]))
		elif data['y'][i] == 8:
			allVectorClass8.append(createLearningVector(data['X'][:, :, :, i]))
		elif data['y'][i] == 9:
			allVectorClass9.append(createLearningVector(data['X'][:, :, :, i]))
		else:
			allVectorClass10.append(createLearningVector(data['X'][:, :, :, i]))
			
	if len(allVectorClass1) != 0:
		avgVector[1] = np.average(allVectorClass1, axis=0)
	if len(allVectorClass2) != 0:
		avgVector[2] = np.average(allVectorClass2, axis=0)
	if len(allVectorClass3) != 0:
		avgVector[3] = np.average(allVectorClass3, axis=0)
	if len(allVectorClass4) != 0:
		avgVector[4] = np.average(allVectorClass4, axis=0)
	if len(allVectorClass5) != 0:
		avgVector[5] = np.average(allVectorClass5, axis=0)
	if len(allVectorClass6) != 0:
		avgVector[6] = np.average(allVectorClass6, axis=0)
	if len(allVectorClass7) != 0:
		avgVector[7] = np.average(allVectorClass7, axis=0)
	if len(allVectorClass8) != 0:
		avgVector[8] = np.average(allVectorClass8, axis=0)
	if len(allVectorClass9) != 0:
		avgVector[9] = np.average(allVectorClass9, axis=0)
	if len(allVectorClass10) != 0:
		avgVector[10] = np.average(allVectorClass10, axis=0)
	print("--averageLearningVector: fin calcul")

	return avgVector

if __name__ == "__main__":
	
	start = time.time()

	print("--Classif.py: Starting learning vector")
	avgVector = averageLearningVector(train_data)

	end = time.time()

	end = end - start
	end = end/60
	print("--Classif.py: Time taken: ")
	print(end)
	
	print("salut 2 :")
	print(avgVector)
