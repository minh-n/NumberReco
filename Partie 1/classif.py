import numpy as np 
import pickle as pi
from scipy.io import loadmat
import time 
import argparse
import sys

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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="automatic picture analysis")
	group = parser.add_mutually_exclusive_group()
	group.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
	group.add_argument("-q", "--quiet", action="store_true", help="reduce output verbosity")
	parser.add_argument("-d", "--data", type=str, help="the .mat file to analyze")
	parser.add_argument("-s", "--size", type=int, help="the size of the data to be analyzed")
	args = parser.parse_args()

	if not args.data:
		raise IOError("Please specify a data.mat with the -d or --data option")

	imageLimit = 100 if not args.size else args.size
	
	try:
		data = loadmat(args.data)

		start = time.time()
		knn_model = pi.load(open('bestModel.sav', 'rb'))
		test = reshapeData(data, imageLimit)
		knn_predictions = knn_model.predict(test['X'])
		accuracy = knn_model.score(test['X'], test['y']) 
		cm = confusion_matrix(test['y'], knn_predictions) 
		end = time.time()
		executionTime = end - start

		print("-----------------\nDisplaying the final results\n-----------------\n")
		print("analyzed images : " + str(imageLimit))
		print("accuracy : " + str(accuracy*100))
		print("execution's time : " + str(executionTime))
		print("confusion matrix : \n" + str(cm))

	except (IOError, OSError) as error:
		sys.stderr.write(error.strerror + "\n")
