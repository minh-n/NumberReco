import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

def reshapeArray(data):
	dict_data = {}
	new_data = []

	for i in range(len(data['y'])):
		nsamples, nx, ny = data['X'][:, :, :, i].shape
		new_data.append(data['X'][:, :, :, i].reshape((nsamples,nx*ny))) 

	dict_data['X'] = new_data
	dict_data['y'] = data['y'].ravel()

	return dict_data


def linearSVM(train, test):
	svm_model_linear = SVC(kernel = 'linear').fit(train['X'], train['y']) 
	svm_predictions = svm_model_linear.predict(test['X']) 

	accuracy = svm_model_linear.score(test['X'], test['y'])
	print(accuracy)

	cm = confusion_matrix(test['y'], svm_predictions) 

	return cm

def kNearestNeighbours(train, test):
	knn = KNeighborsClassifier().fit(train['X'], train['y']) 
  
	knn_predictions = knn.predict(test['X'])
	accuracy = knn.score(test['X'], test['y']) 
	print(accuracy)
	 
	cm = confusion_matrix(test['y'], knn_predictions) 

	return cm

def classifier(train, test, classif):
	new_train = reshapeArray(train)
	new_test = reshapeArray(test)

	classif(train, test)

if __name__ == "__main__":
	classifier(train_data, test_data, linearSVM)