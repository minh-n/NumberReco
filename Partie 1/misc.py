import numpy as np
import time as time 
import pickle as pi
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

# ---------------------------------
# reduce and flatten the size of the data
def reshapeData(data, size):
	dict_data = {}
	new_data = []

	size = size if size < len(data['y']) else len(data['y'])

	if len(data['X'].shape) == 4: 
		for i in range(size):
			new_data.append(data['X'][:, :, :, i].ravel())

		array = np.array(new_data)
		label = np.array(data['y'].ravel())

		dict_data['X'] = array
		dict_data['y'] = label[:size]

	else:
		dict_data['X'] = data['X'][:size]
		dict_data['y'] = data['y'][:size]

	return dict_data

# ---------------------------------
# apply the pca to the data
# nc represents the final number of compenents
def pca(nc, data, imageLimit):
	dict_data = reshapeData(data, imageLimit)
	pca = PCA(n_components=nc)
	dict_data['X'] = pca.fit_transform(dict_data['X'])

	return dict_data

# ---------------------------------
# apply the pca to two data
def pcaWithTwoData(nc, train, test, imageLimit):
	start = time.time()

	dict_train = pca(nc, train, imageLimit)
	dict_test = pca(nc, test, imageLimit)

	end = time.time()
	executionTime = end - start

	return dict_train, dict_test, executionTime

# ---------------------------------
# learning function : appends every images into separate lists
# and computes the average of the class
def averageLearningVector(data, imageLimit):
	avgVector = {}
	allClassVectors = [[] for i in range(10)] 
	imageLimit = imageLimit if imageLimit < len(data["y"]) else len(data["y"])

	for i in range(imageLimit):
		if len(data['X'].shape) == 4:
			allClassVectors[data['y'][i][0]-1].append(data['X'][:, :, :, i])
		else:
			allClassVectors[data['y'][i]-1].append(data['X'][i])

	for i in range(10):
		if len(allClassVectors[i]) != 0:
			avgVector[i+1] = np.average(allClassVectors[i], axis=0)

	return avgVector


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
def minimumDistanceClassifier(train_data, test_data, imageLimit):
	successTotal = 0
	successPerClass = [0]*11
	numberOfImagePerClass = [0]*11
	imageLimit = imageLimit if imageLimit < len(test_data["y"]) else len(test_data["y"])

	start = time.time()

	avgVector = averageLearningVector(train_data, imageLimit)

	if len(test_data['X'].shape) == 4:
		for i in range(imageLimit):
			label = findLabel(test_data["X"][:, :, :, i], avgVector)
			numberOfImagePerClass[label] += 1
			if label == test_data["y"][i][0]:
				successTotal += 1
				successPerClass[label] += 1
	else:
		for i in range(imageLimit):
			label = findLabel(test_data["X"][i], avgVector)
			numberOfImagePerClass[label] += 1
			if label == test_data["y"][i]:
				successTotal += 1
				successPerClass[label] += 1

	successPerClass[0] = successPerClass[10]
	numberOfImagePerClass[0] = numberOfImagePerClass[10]

	end = time.time()
	executionTime = end - start

	return successTotal, successPerClass, numberOfImagePerClass, executionTime

# ---------------------------------
# Support Vector Machine with linear kernel
def linearSVM(train, test, imageLimit=None):
	start = time.time()

	if imageLimit != None:
		train = reshapeData(train, imageLimit)

	svm_model_linear = LinearSVC().fit(train['X'], train['y']) 
	svm_predictions = svm_model_linear.predict(test['X']) 
	accuracy = svm_model_linear.score(test['X'], test['y'])
	cm = confusion_matrix(test['y'], svm_predictions) 

	end = time.time()
	executionTime = end - start

	return cm, accuracy, executionTime

# ---------------------------------
# Classifier implementing the k-nearest neighbors vote
def kNearestNeighbours(train, test, imageLimit=None, neighbours=5):
	start = time.time()

	if imageLimit != None:
		train = reshapeData(train, imageLimit)

	knn = KNeighborsClassifier(n_neighbors=neighbours).fit(train['X'], train['y']) 
	knn_predictions = knn.predict(test['X'])
	accuracy = knn.score(test['X'], test['y']) 
	cm = confusion_matrix(test['y'], knn_predictions) 

	end = time.time()
	executionTime = end - start

	return cm, accuracy, executionTime

def savekNearestNeighboursWithPickle(train, imageLimit=None, neighbours=5):
	if imageLimit != None:
		train = reshapeData(train, imageLimit)
	else:
		train = reshapeData(train, len(train['y']))

	knn = KNeighborsClassifier(n_neighbors=neighbours).fit(train['X'], train['y']) 
	pi.dump(knn, open('bestModel.sav', 'wb'))

# ---------------------------------
# function to apply the pca and the mdc to the data
def pcaAndMinimumDistanceClassifier(train_data, test_data, imageLimit, nc):
	# train, test, timePCA = pcaWithTwoData(nc, train_data, test_data, imageLimit)
	successTotal, successPerClass, numberOfImagePerClass, timeMDC = minimumDistanceClassifier(train_data, test_data, imageLimit)

	return successTotal, successPerClass, numberOfImagePerClass, timeMDC

# ---------------------------------
# function to apply the pca and a sklearn classifier to the data
def pcaAndSklearn(train_data, test_data, imageLimit, nc, sklearn, neighbours=5):
	# train, test, timePCA = pcaWithTwoData(nc, train_data, test_data, imageLimit)
	cm, accuracy, timeSklearn = sklearn(train_data, test_data, imageLimit, neighbours) if neighbours != 5 else sklearn(train_data, test_data, imageLimit)

	return cm, accuracy, timeSklearn

# ---------------------------------
# function to save the results in a file.
# fct = 0 for mdc, 1 = svc and 2 = knn
def writeFilePCA(timePCA, fileName, train_data, test_data, imageLimit, nc, fct, neighbours=5):
	print("\nStarting to write " + fileName + "...")
	with open(fileName, "w") as file:
		if fct == 0:
			successTotal, successPerClass, numberOfImagePerClass, timeMDC = pcaAndMinimumDistanceClassifier(train_data, test_data, imageLimit, nc)

			file.write("-----------------\nDisplaying the final results\n-----------------\n")
			for i in range(10):
				successPercentage =  100.*successPerClass[i]/numberOfImagePerClass[i] 
				file.write("\n---Success rate for " + str(i) + ": " + str(successPerClass[i]) + " / " 
							+ str(numberOfImagePerClass[i]) + " (" + str(successPercentage) + "%)")

			successPercentage =  100.*successTotal/imageLimit
			file.write("\nTotal success rate : " + str(successTotal) + " / " + str(imageLimit) + " (" 
						+ str(successPercentage) + "%)")

			file.write("\nPCA's execution time : " + str(timePCA))
			file.write("\nMDC's execution time : " + str(timeMDC))
			file.write("\nTotal time : " + str(timePCA + timeMDC))

		elif fct == 1:
			cm, accuracy, timeSklearn = pcaAndSklearn(train_data, test_data, imageLimit, nc, linearSVM)

			file.write("-----------------\nDisplaying the final results\n-----------------\n")
			file.write("\nConfusion matrix :\n")
			file.write(str(cm))
			file.write("\n accuracy : " + str(accuracy))
			file.write("\nPCA's execution time : " + str(timePCA))
			file.write("\nLinearSVM's execution time : " + str(timeSklearn))
			file.write("\nTotal time : " + str(timePCA + timeSklearn))

		else:
			cm, accuracy, timeSklearn = pcaAndSklearn(train_data, test_data, imageLimit, nc, kNearestNeighbours, neighbours)

			file.write("-----------------\nDisplaying the final results\n-----------------\n")
			file.write("\nConfusion matrix :\n")
			file.write(str(cm))
			file.write("\n accuracy : " + str(accuracy))
			file.write("\nPCA's execution time : " + str(timePCA))
			file.write("\nKNN's execution time : " + str(timeSklearn))
			file.write("\nTotal time : " + str(timePCA + timeSklearn))

# ---------------------------------
# function to apply a sklearn classifier to the data
def Sklearn(train_data, test_data, imageLimit, sk, neighbours=5):
	train = reshapeData(train_data, imageLimit)
	test = reshapeData(test_data, imageLimit)
	return sk(train, test, imageLimit, neighbours) if neighbours != 5 else sk(train, test, imageLimit)

# ---------------------------------
# function to save the results in a file.
# 1 = svc and 2 = knn
def writeFile(fileName, train_data, test_data, imageLimit, fct, neighbours=5):
	print("\nStarting to write " + fileName + "...")
	with open(fileName, "w") as file:
		if fct == 1:
			cm, accuracy, timeSklearn = Sklearn(train_data, test_data, imageLimit, linearSVM)

			file.write("-----------------\nDisplaying the final results\n-----------------\n")
			file.write("\nConfusion matrix :\n")
			file.write(str(cm))
			file.write("\n accuracy : " + str(accuracy))
			file.write("\nLinearSVM's execution time : " + str(timeSklearn))

		else:
			cm, accuracy, timeSklearn = Sklearn(train_data, test_data, imageLimit, kNearestNeighbours, neighbours)

			file.write("-----------------\nDisplaying the final results\n-----------------\n")
			file.write("\nConfusion matrix :\n")
			file.write(str(cm))
			file.write("\n accuracy : " + str(accuracy))
			file.write("\nKNN's execution time : " + str(timeSklearn))


if __name__ == "__main__":
	train_data = loadmat('train_32x32.mat')
	test_data = loadmat('test_32x32.mat')

	prepro_train_data = loadmat('prepro_train.mat')
	prepro_test_data = loadmat('prepro_test.mat')

	# savekNearestNeighboursWithPickle(train_data)

	# PCA

	# print("\nPCA :\nPCA 50%")
	# imageLimit = train_data['y'].shape[0]
	# nc = int((train_data['X'].shape[0] * train_data['X'].shape[1] * train_data['X'].shape[2]) * 0.5)
	# tmp1, tmp2, tmp3 = pcaWithTwoData(nc, train_data, test_data, imageLimit)

	# tmp = (tmp1, tmp2, tmp3)
	# pi.dump(tmp, open('pickle_pca50.sav', 'wb'))

	# print("\nPCA 75%")
	# nc = int((train_data['X'].shape[0] * train_data['X'].shape[1] * train_data['X'].shape[2]) * 0.75)
	# tmp1, tmp2, tmp3 = pcaWithTwoData(nc, train_data, test_data, imageLimit)

	# tmp = (tmp1, tmp2, tmp3)
	# pi.dump(tmp, open('pickle_pca75.sav', 'wb'))
	

	# print("\nPCA 95%")
	# nc = int((train_data['X'].shape[0] * train_data['X'].shape[1] * train_data['X'].shape[2]) * 0.95)
	# tmp1, tmp2, tmp3 = pcaWithTwoData(nc, train_data, test_data, imageLimit)

	# tmp = (tmp1, tmp2, tmp3)
	# pi.dump(tmp, open('pickle_pca95.sav', 'wb'))


	# Preprocessing + PCA


	# print("\nPCA :\nPreprocessing + PCA 50%")
	# nc = int((prepro_train_data['X'].shape[0] * prepro_train_data['X'].shape[1] * prepro_train_data['X'].shape[2]) * 0.5)
	# tmp1, tmp2, tmp3 = pcaWithTwoData(nc, prepro_train_data, prepro_test_data, imageLimit)

	# tmp = (tmp1, tmp2, tmp3)
	# pi.dump(tmp, open('prepro_pickle_pca50.sav', 'wb'))

	# print("\nPreprocessing + PCA 75%")
	# nc = int((prepro_train_data['X'].shape[0] * prepro_train_data['X'].shape[1] * prepro_train_data['X'].shape[2]) * 0.75)
	# tmp1, tmp2, tmp3 = pcaWithTwoData(nc, prepro_train_data, prepro_test_data, imageLimit)

	# tmp = (tmp1, tmp2, tmp3)
	# pi.dump(tmp, open('prepro_pickle_pca75.sav', 'wb'))

	# print("\nPreprocessing + PCA 95%")
	# nc = int((prepro_train_data['X'].shape[0] * prepro_train_data['X'].shape[1] * prepro_train_data['X'].shape[2]) * 0.95)
	# tmp1, tmp2, tmp3 = pcaWithTwoData(nc, prepro_train_data, prepro_test_data, imageLimit)

	# tmp = (tmp1, tmp2, tmp3)
	# pi.dump(tmp, open('prepro_pickle_pca95.sav', 'wb'))


	# Loading PCA


	pca50_train, pca50_test, timePCA50 = pi.load(open('pickle_pca50.sav', 'rb'))
	pca75_train, pca75_test, timePCA75 = pi.load(open('pickle_pca75.sav', 'rb'))
	pca95_train, pca95_test, timePCA95 = pi.load(open('pickle_pca95.sav', 'rb'))

	prepro_pca50_train, prepro_pca50_test, prepro_timePCA50 = pi.load(open('prepro_pickle_pca50.sav', 'rb'))
	prepro_pca75_train, prepro_pca75_test, prepro_timePCA75 = pi.load(open('prepro_pickle_pca75.sav', 'rb'))
	prepro_pca95_train, prepro_pca95_test, prepro_timePCA95 = pi.load(open('prepro_pickle_pca95.sav', 'rb'))


	# PCA + MDC

	imageLimit = train_data['y'].shape[0]
	nc = 0

	# print("\nPCA 10%")
	# nc = int((train_data['X'].shape[0] * train_data['X'].shape[1] * train_data['X'].shape[2]) * 0.10)
	# pca10_train, pca10_test, timePCA10 = pcaWithTwoData(nc, train_data, test_data, imageLimit)
	# writeFilePCA(timePCA10, "pca10_mdc", pca10_train, pca10_test, imageLimit, nc, 0)

	# print("\nPCA 20%")
	# nc = int((train_data['X'].shape[0] * train_data['X'].shape[1] * train_data['X'].shape[2]) * 0.20)
	# pca20_train, pca20_test, timePCA20 = pcaWithTwoData(nc, train_data, test_data, imageLimit)
	# writeFilePCA(timePCA20, "pca20_mdc", pca20_train, pca20_test, imageLimit, nc, 0)

	# print("\nPCA 30%")
	# nc = int((train_data['X'].shape[0] * train_data['X'].shape[1] * train_data['X'].shape[2]) * 0.30)
	# pca30_train, pca30_test, timePCA30 = pcaWithTwoData(nc, train_data, test_data, imageLimit)
	# writeFilePCA(timePCA30, "pca30_mdc", pca30_train, pca30_test, imageLimit, nc, 0)

	# print("\nPCA 40%")
	# nc = int((train_data['X'].shape[0] * train_data['X'].shape[1] * train_data['X'].shape[2]) * 0.40)
	# pca40_train, pca40_test, timePCA40 = pcaWithTwoData(nc, train_data, test_data, imageLimit)
	# writeFilePCA(timePCA40, "pca40_mdc", pca40_train, pca40_test, imageLimit, nc, 0)

	# print("\nPCA 90%")
	# nc = int((train_data['X'].shape[0] * train_data['X'].shape[1] * train_data['X'].shape[2]) * 0.90)
	# pca90_train, pca90_test, timePCA90 = pcaWithTwoData(nc, train_data, test_data, imageLimit)
	# writeFilePCA(timePCA90, "pca90_mdc", pca90_train, pca90_test, imageLimit, nc, 0)

	writeFilePCA(timePCA50, "pca50_mdc", pca50_train, pca50_test, imageLimit, nc, 0)
	writeFilePCA(prepro_timePCA50, "prepo_pca50_mdc", prepro_pca50_train, prepro_pca50_test, imageLimit, nc, 0)

	writeFilePCA(timePCA75, "pca75_mdc", pca75_train, pca75_test, imageLimit, nc, 0)
	writeFilePCA(prepro_timePCA75, "prepo_pca75_mdc", prepro_pca75_train, prepro_pca75_test, imageLimit, nc, 0)

	writeFilePCA(timePCA95, "pca95_mdc", pca95_train, pca95_test, imageLimit, nc, 0)
	writeFilePCA(prepro_timePCA95, "prepo_pca95_mdc", prepro_pca95_train, prepro_pca95_test, imageLimit, nc, 0)


	# PCA + LinearSVM

	imageLimit = 1000
	pca50_train_1000 = reshapeData(pca50_train, imageLimit)
	pca50_test_1000 = reshapeData(pca50_test, imageLimit)
	writeFilePCA(timePCA50, "pca50_LinearSVM_1000", pca50_train_1000, pca50_test_1000, imageLimit, nc, 1)

	prepro_pca50_train_1000 = reshapeData(prepro_pca50_train, imageLimit)
	prepro_pca50_test_1000 = reshapeData(prepro_pca50_test, imageLimit)
	writeFilePCA(prepro_timePCA50, "prepro_pca50_LinearSVM_1000", prepro_pca50_train_1000, prepro_pca50_test_1000, imageLimit, nc, 1)

	pca75_train_1000 = reshapeData(pca75_train, imageLimit)
	pca75_test_1000 = reshapeData(pca75_test, imageLimit)
	writeFilePCA(timePCA75, "pca75_LinearSVM_1000", pca75_train_1000, pca75_test_1000, imageLimit, nc, 1)

	prepro_pca75_train_1000 = reshapeData(prepro_pca75_train, imageLimit)
	prepro_pca75_test_1000 = reshapeData(prepro_pca75_test, imageLimit)
	writeFilePCA(prepro_timePCA75, "prepro_pca75_LinearSVM_1000", prepro_pca75_train_1000, prepro_pca75_test_1000, imageLimit, nc, 1)

	pca95_train_1000 = reshapeData(pca95_train, imageLimit)
	pca95_test_1000 = reshapeData(pca95_test, imageLimit)
	writeFilePCA(timePCA95, "pca95_LinearSVM_1000", pca95_train_1000, pca95_test_1000, imageLimit, nc, 1)

	prepro_pca95_train_1000 = reshapeData(prepro_pca95_train, imageLimit)
	prepro_pca95_test_1000 = reshapeData(prepro_pca95_test, imageLimit)
	writeFilePCA(prepro_timePCA95, "prepro_pca95_LinearSVM_1000", prepro_pca95_train_1000, prepro_pca95_test_1000, imageLimit, nc, 1)

	imageLimit = 2000
	pca75_train_2000 = reshapeData(pca75_train, imageLimit)
	pca75_test_2000 = reshapeData(pca75_test, imageLimit)
	writeFilePCA(timePCA75, "pca75_LinearSVM_2000", pca75_train_2000, pca75_test_2000, imageLimit, nc, 1)

	prepro_pca75_train_2000 = reshapeData(prepro_pca75_train, imageLimit)
	prepro_pca75_test_2000 = reshapeData(prepro_pca75_test, imageLimit)
	writeFilePCA(prepro_timePCA75, "prepro_pca75_LinearSVM_2000", prepro_pca75_train_2000, prepro_pca75_test_2000, imageLimit, nc, 1)

	imageLimit = 5000
	pca75_train_5000 = reshapeData(pca75_train, imageLimit)
	pca75_test_5000 = reshapeData(pca75_test, imageLimit)
	writeFilePCA(timePCA75, "pca75_LinearSVM_5000", pca75_train_5000, pca75_test_5000, imageLimit, nc, 1)

	prepro_pca75_train_5000 = reshapeData(prepro_pca75_train, imageLimit)
	prepro_pca75_test_5000 = reshapeData(prepro_pca75_test, imageLimit)
	writeFilePCA(prepro_timePCA75, "prepro_pca75_LinearSVM_5000", prepro_pca75_train_5000, prepro_pca75_test_5000, imageLimit, nc, 1)


	# PCA + KNN

	imageLimit = 1000
	writeFilePCA(timePCA50, "pca50_KNN5_1000", pca50_train_1000, pca50_test_1000, imageLimit, nc, 2)
	writeFilePCA(prepro_timePCA50, "prepro_pca50_KNN5_1000", prepro_pca50_train_1000, prepro_pca50_test_1000, imageLimit, nc, 2)

	writeFilePCA(timePCA75, "pca75_KNN10_1000", pca75_train_1000, pca75_test_1000, imageLimit, nc, 2, 10)
	writeFilePCA(prepro_timePCA75, "prepro_pca75_KNN10_1000", prepro_pca75_train_1000, prepro_pca75_test_1000, imageLimit, nc, 2, 10)

	writeFilePCA(timePCA95, "pca95_KNN20_1000", pca95_train_1000, pca95_test_1000, imageLimit, nc, 2, 20)
	writeFilePCA(prepro_timePCA95, "prepro_pca95_KNN20_1000", prepro_pca95_train_1000, prepro_pca95_test_1000, imageLimit, nc, 2, 20)

	imageLimit = 2000
	writeFilePCA(timePCA75, "pca75_KNN10_2000", pca75_train_2000, pca75_test_2000, imageLimit, nc, 2)
	writeFilePCA(prepro_timePCA75, "prepro_pca75_KNN10_2000", prepro_pca75_train_2000, prepro_pca75_test_2000, imageLimit, nc, 2)


	# LinearSVM

	imageLimit = 1000
	writeFile("LinearSVM_1000", train_data, test_data, imageLimit, 1)
	writeFile("prepro_LinearSVM_1000", prepro_train_data, prepro_test_data, imageLimit, 1)

	imageLimit = 1000
	writeFile("LinearSVM_1000", train_data, test_data, imageLimit, 1)
	writeFile("prepro_LinearSVM_1000", prepro_train_data, prepro_test_data, imageLimit, 1)	

	imageLimit = 2000
	writeFile("LinearSVM_2000", train_data, test_data, imageLimit, 1)
	writeFile("prepro_LinearSVM_2000", prepro_train_data, prepro_test_data, imageLimit, 1)

	# KNN 

	imageLimit = 1000
	writeFile("KNN5_1000", train_data, test_data, imageLimit, 2)
	writeFile("prepro_KNN5_1000", prepro_train_data, prepro_test_data, imageLimit, 2)

	writeFile("KNN10_1000", train_data, test_data, imageLimit, 2, 10)
	writeFile("prepro_KNN10_1000", prepro_train_data, prepro_test_data, imageLimit, 2, 10)

	writeFile("KNN20_1000", train_data, test_data, imageLimit, 2, 20)
	writeFile("prepro_KNN20_1000", prepro_train_data, prepro_test_data, imageLimit, 2, 20)

	imageLimit = 2000
	writeFile("KNN5_2000", train_data, test_data, imageLimit, 2)
	writeFile("prepro_KNN5_2000", prepro_train_data, prepro_test_data, imageLimit, 2)


	# SIZE 5,000

	imageLimit = 5000
	writeFile("LinearSVM_5000", train_data, test_data, imageLimit, 1)
	writeFile("prepro_LinearSVM_5000", prepro_train_data, prepro_test_data, imageLimit, 1)

	writeFile("KNN5_5000", train_data, test_data, imageLimit, 2)
	writeFile("prepro_KNN5_5000", prepro_train_data, prepro_test_data, imageLimit, 2)


	# SIZE OVER 9,000

	imageLimit = 10000
	writeFile("LinearSVM_10000", train_data, test_data, imageLimit, 1)
	writeFile("prepro_LinearSVM_10000", prepro_train_data, prepro_test_data, imageLimit, 1)
	writeFile("KNN5_10000", train_data, test_data, imageLimit, 2)
	writeFile("prepro_KNN5_10000", prepro_train_data, prepro_test_data, imageLimit, 2)

	pca95_train_10000 = reshapeData(pca95_train, imageLimit)
	pca95_test_10000 = reshapeData(pca95_test, imageLimit)
	prepro_pca95_train_10000 = reshapeData(prepro_pca95_train, imageLimit)
	prepro_pca95_test_10000 = reshapeData(prepro_pca95_test, imageLimit)

	writeFilePCA(timePCA95, "pca95_LinearSVM_10000", pca95_train_10000, pca95_test_10000, imageLimit, nc, 1)
	writeFilePCA(prepro_timePCA95, "prepro_pca95_LinearSVM_10000", prepro_pca95_train_10000, prepro_pca95_test_10000, imageLimit, nc, 1)
	writeFilePCA(timePCA95, "pca95_KNN5_10000", pca95_train_10000, pca95_test_10000, imageLimit, nc, 2)
	writeFilePCA(prepro_timePCA95, "prepro_pca95_KNN5_10000", prepro_pca95_train_10000, prepro_pca95_test_10000, imageLimit, nc, 2)

	imageLimit = 20000
	writeFile("LinearSVM_20000", train_data, test_data, imageLimit, 1)
	writeFile("prepro_LinearSVM_20000", prepro_train_data, prepro_test_data, imageLimit, 1)
	writeFile("KNN5_20000", train_data, test_data, imageLimit, 2)
	writeFile("prepro_KNN5_20000", prepro_train_data, prepro_test_data, imageLimit, 2)

	pca95_train_20000 = reshapeData(pca95_train, imageLimit)
	pca95_test_20000 = reshapeData(pca95_test, imageLimit)
	prepro_pca95_train_20000 = reshapeData(prepro_pca95_train, imageLimit)
	prepro_pca95_test_20000 = reshapeData(prepro_pca95_test, imageLimit)

	writeFilePCA(timePCA95, "pca95_LinearSVM_20000", pca95_train_20000, pca95_test_20000, imageLimit, nc, 1)
	writeFilePCA(prepro_timePCA95, "prepro_pca95_LinearSVM_20000", prepro_pca95_train_20000, prepro_pca95_test_20000, imageLimit, nc, 1)
	writeFilePCA(timePCA95, "pca95_KNN5_20000", pca95_train_20000, pca95_test_20000, imageLimit, nc, 2)
	writeFilePCA(prepro_timePCA95, "prepro_pca95_KNN5_20000", prepro_pca95_train_20000, prepro_pca95_test_20000, imageLimit, nc, 2)

	imageLimit = 30000
	writeFile("LinearSVM_30000", train_data, test_data, imageLimit, 1)
	writeFile("prepro_LinearSVM_30000", prepro_train_data, prepro_test_data, imageLimit, 1)
	writeFile("KNN5_30000", train_data, test_data, imageLimit, 2)
	writeFile("prepro_KNN5_30000", prepro_train_data, prepro_test_data, imageLimit, 2)

	pca95_train_30000 = reshapeData(pca95_train, imageLimit)
	pca95_test_30000 = reshapeData(pca95_test, imageLimit)
	prepro_pca95_train_30000 = reshapeData(prepro_pca95_train, imageLimit)
	prepro_pca95_test_30000 = reshapeData(prepro_pca95_test, imageLimit)

	writeFilePCA(timePCA95, "pca95_LinearSVM_30000", pca95_train_30000, pca95_test_30000, imageLimit, nc, 1)
	writeFilePCA(prepro_timePCA95, "prepro_pca95_LinearSVM_30000", prepro_pca95_train_30000, prepro_pca95_test_30000, imageLimit, nc, 1)
	writeFilePCA(timePCA95, "pca95_KNN5_30000", pca95_train_30000, pca95_test_30000, imageLimit, nc, 2)
	writeFilePCA(prepro_timePCA95, "prepro_pca95_KNN5_30000", prepro_pca95_train_30000, prepro_pca95_test_30000, imageLimit, nc, 2)