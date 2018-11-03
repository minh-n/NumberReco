import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

train_data = loadmat('train_32x32.mat')
# test_data = loadmat('test_32x32.mat')

for i in range(len(train_data['y'])):
	print(train_data['y'][i])

# print(train_data['X'][1])

#plt.imshow(test_data['X'][:, :, :, image_idx])
#plt.show()