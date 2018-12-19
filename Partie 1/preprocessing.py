import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal

train_data = loadmat('train_32x32.mat')
#test_data = loadmat('test_32x32.mat')

def negative(picture):
	(lines, columns, colors) = np.shape(picture)

	for i in range(lines):
		for j in range(columns):
			for c in range(colors):
				picture[i, j, c] = 255 - picture[i, j, c]

def grey(picture):
	(lines, columns, colors) = np.shape(picture)

	for i in range(lines):
		for j in range(columns):
			grey = 0.299*picture[i, j, 0] + 0.587*picture[i, j, 1] + 0.114*picture[i, j, 2]
			picture[i, j, :] = [grey for i in range(colors)]

def avg_color(mat):
	lines, columns, _ = mat.shape
	avg = 0
	for i in range (lines):
		for j in range(columns):
			avg += int(np.mean(mat[i][j]))
	avg /= (lines*columns)
	return avg

def thresholdingV2(picture):
	(lines, columns, colors) = np.shape(picture)

	avg = avg_color(picture)

	for i in range(lines):
		for j in range(columns):
			for c in range(colors):
				picture[i, j, c] = 255 if picture[i, j, c] > avg else 0

def thresholding(picture):
	(lines, columns, colors) = np.shape(picture)

	for i in range(lines):
		for j in range(columns):
			for c in range(colors):
				picture[i, j, c] = 255 if picture[i, j, c] > 127 else 0

def lowPassFilter(picture):
	colors = np.shape(picture)[2]
	kernel = [[ 1,  1,  1],
			  [ 1,  6,  1],
			  [ 1,  1,  1]]

	for c in range(colors):
		picture[:, :, c] = signal.convolve2d(picture[:, :, c], kernel, boundary='symm', mode='same')

def highPassFilter(picture):
	colors = np.shape(picture)[2]
	kernel = [[ 0,  -4,  0],
			  [-4,  17, -4],
			  [ 0,  -4,  0]]

	for c in range(colors):
		picture[:, :, c] = signal.convolve2d(picture[:, :, c], kernel, boundary='symm', mode='same')

def gradient(picture):
	colors = np.shape(picture)[2]
	kernel = np.array([[ -3-3j, 0-10j,  +3-3j],
                       [-10+0j, 0+ 0j, +10+0j],
                       [ -3+3j, 0+10j,  +3+3j]])

	for c in range(colors):
		picture[:, :, c] = signal.convolve2d(picture[:, :, c], kernel,  boundary='symm', mode='same')

def sobelFilter(picture):
	colors = np.shape(picture)[2]
	kernel1 = [[ 1,  0,  -1],
			   [ 2,  0,  -2],
			   [ 1,  0,  -1]]

	kernel2 = [[  1,   2,   1],
			   [  0,   0,   0],
			   [ -1,  -2,  -1]]

	for c in range(colors):
		picture[:, :, c] = signal.convolve2d(picture[:, :, c], kernel1,  boundary='symm', mode='same')

	for c in range(colors):
		picture[:, :, c] = signal.convolve2d(picture[:, :, c], kernel2,  boundary='symm', mode='same')

def brightness(picture, offset):
	(lines, columns, colors) = np.shape(picture)

	for i in range(lines):
		for j in range(columns):
			for c in range(colors):
				picture[i, j, c] = max(0, min(255, picture[i, j, c] + offset))

def contrast(picture):
	(lines, columns, colors) = np.shape(picture)

	for i in range(lines):
		for j in range(columns):
			for c in range(colors):
				picture[i, j, c] = 0 if picture[i, j, c] < 50 else (255 if picture[i, j, c] > 225 else ((255/195)*(picture[i, j, c] - 50) + 0.5))

def sum(histogram, k):
	result = 0
	for i in range(k):
		result += histogram[i]
	return result

def histogramEqualization(picture):
	(lines, columns, colors) = np.shape(picture)
	histogram = np.zeros(256)
	grey(picture)

	for i in range(lines):
		for j in range(columns):
			for c in range(colors):
				histogram[picture[i, j, c]] += 1

	constant = (256 - 1)/(lines * columns)
	for i in range(lines):
		for j in range(columns):
			for c in range(colors):
				picture[i, j, c] = constant * sum(histogram, picture[i, j, c])

def imageProcessing(data, filter, setting=None):
	print("--preprocessing.py: processing dataset with {}".format(filter))
	if setting is None:
		for i in range(len(data['y'])):
			print("Processing {}...".format(i))
			filter(data['X'][:, :, :, i])
	else:
		for i in  range(len(data['y'])):
			#print("Processing {}...".format(i))
			filter(data['X'][:, :, :, i], setting)


def imageProcessingTwoFilters(data, filter1, filter2, imageLimit, setting=None):
	print("--preprocessing.py: processing dataset with {} and {}".format(str(filter1), str(filter2)))
	
	if(len(data["y"]) < imageLimit):
		imageLimit = len(data["y"]) 

	if setting is None:
		for i in range(imageLimit):
			#print("Processing {}...".format(i))
			filter1(data['X'][:, :, :, i])
			filter2(data['X'][:, :, :, i])
	else:
		for i in range(imageLimit):
			#print("Processing {}...".format(i))
			filter1(data['X'][:, :, :, i], setting)
			filter2(data['X'][:, :, :, i])
	return data

def imageProcessingThreeFilters(data, filter1, filter2, filter3, imageLimit, setting=None):
	print("--preprocessing.py: processing dataset with {} and {}".format(str(filter1), str(filter2)))
	
	if(len(data["y"]) < imageLimit):
		imageLimit = len(data["y"]) 

	if setting is None:
		for i in range(imageLimit):
			#print("Processing {}...".format(i))
			filter1(data['X'][:, :, :, i])
			filter2(data['X'][:, :, :, i])
			filter3(data['X'][:, :, :, i])

	else:
		for i in range(imageLimit):
			#print("Processing {}...".format(i))
			filter1(data['X'][:, :, :, i])
			filter2(data['X'][:, :, :, i])
			filter3(data['X'][:, :, :, i], setting)

	return data

if __name__ == "__main__":
	
	index = 1909
	plt.imshow(train_data['X'][:, :, :, index])
	plt.show()
	# negative(train_data['X'][:, :, :, index])
	# grey(train_data['X'][:, :, :, index])
	# thresholding(train_data['X'][:, :, :, index])
	lowPassFilter(train_data['X'][:, :, :, index])
	highPassFilter(train_data['X'][:, :, :, index]) #messy pixels
	# gradient(train_data['X'][:, :, :, index])
	# sobelFilter(train_data['X'][:, :, :, index])  #messy pixels
	# brightness(train_data['X'][:, :, :, index], 60)
	#contrast(train_data['X'][:, :, :, index]) 
	# imageProcessing(train_data, contrast)
	# histogramEqualization(train_data['X'][:, :, :, index])
	plt.imshow(train_data['X'][:, :, :, index])
	plt.show()




