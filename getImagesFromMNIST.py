from mnist import MNIST
import random
import numpy as np
import pickle

RAW_DATA_DIR = 'data/ubyte'
PARSED_IMAGE_DIR = 'data/'
DATA_PICKLE_NAME = 'data/pickles/test_data.pickle'
LABELS_PICKLE_NAME = 'data/pickles/test_labels.pickle'
mndata = MNIST(RAW_DATA_DIR)

images, labels = mndata.load_testing()

data = []

for i in range(0, len(images)):
	ubyte_img = mndata.display(images[i]).split('\n')[1:]
	image_as_list = []
	for j in range(0, len(ubyte_img)):
		img_line_list = []
		ubyte_img_line = ubyte_img[j]		
		for k in range(0, len(ubyte_img_line)):
			if(ubyte_img_line[k] == '.'):
				# img_line_list.append(0)
				image_as_list.append(0)
			else:
				# img_line_list.append(255)
				image_as_list.append(255)
		# image_as_list.append(img_line_list)
	image_as_np = np.reshape(np.array(image_as_list), (28,28,1))
	data.append(image_as_list)
data_np = np.array(data)
labels_np = np.array(labels)

#pickle the data
with open(DATA_PICKLE_NAME, 'wb') as f:
	pickle.dump(data_np, f)
print("Data Pickle Created at path : {}".format(DATA_PICKLE_NAME))

with open(LABELS_PICKLE_NAME, 'wb') as f:
	pickle.dump(labels_np, f)
print("Labels Pickle Created at path : {}".format(LABELS_PICKLE_NAME))