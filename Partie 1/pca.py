import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

def pca(variance, data):
	dict_data = {}
	new_data = []

	pca = PCA(variance)
	for i in range(len(data['y'])):
		nsamples, nx, ny = data['X'][:, :, :, i].shape
		tmp = data['X'][:, :, :, i].reshape((nsamples,nx*ny))
		pca.fit(tmp) 
		new_data.append(pca.transform(tmp))

	dict_data['X'] = new_data
	dict_data['y'] = data['y']

	return dict_data

def pcaWithTwoData(variance, train, test):
	dict_train = pca(variance, train)
	dict_test = pca(variance, test)

	return dict_train, dict_test

if __name__ == "__main__":
	train, test = pcaWithTwoData(.95, train_data, test_data)

	print(train['X'][0])