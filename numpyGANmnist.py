import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import cv2
# An implementation of Adversarial Training in GANs on the MNIST dataset using numpy

#0. load in the data done
#1.	define the activation functions done
#2. define the models done
#3. define the forward prop done
#4. define backprop
#5. define the epoch loop
#6. within the loop, generate negatives for the discriminator
#7. train the discriminator
#8. freezing the discriminator, 
#	run end-to-end and train weights of the generator with 'validity' as predicted by the discriminator as the prediction, 
#	input as an n-dim random noise vector and 'valid' as the label for all input samples.

# *using the architecture similar to the keras implementation.

TRAIN_PATH = 'data/pickles/train_data.pickle'
TEST_PATH = 'data/pickles/test_data.pickle'

# TRAIN_LABELS_PATH = 'data/pickles/train_labels.pickle'
# TEST_LABELS_PATH = 'data/pickles/test_labels.pickle'

train_data = []
# train_labels = []
test_data = []
# test_labels = []

with open(TRAIN_PATH,'rb') as f:
	train_data = pickle.load(f).T

with open(TEST_PATH,'rb') as f:
	test_data = pickle.load(f).T

print "Data Loaded: TRAIN data :{}\t TEST data :{}".format(train_data.shape, test_data.shape)
allData = np.hstack([train_data, test_data])
allData[allData > 0] = 1.

batch_size = 1
LR = 1
VAL_SPLIT = 0.05
N_EPOCHS = 200
BN_epsilon = 1e-5 #epsilon for batch norm
smallValue = 1e-25
#discriminator specs:
input_dim = 784
d_nhl1 = 512
d_nhl2 = 256
d_out = 1


#generator specs:
noise_dim = 100
g_nhl1 = 256
g_nhl2 = 512
g_nhl3 = 1024
g_out = 784


#defining the discriminator weights:
d = dict()

d['Whl1'] = np.random.rand(d_nhl1, input_dim) #512x784
d['bhl1'] = np.random.rand(d_nhl1, 1) #512x1

d['Whl2'] = np.random.rand(d_nhl2, d_nhl1) #256x512
d['bhl2'] = np.random.rand(d_nhl2, 1) #256x1

d['Wout'] = np.random.rand(d_out, d_nhl2) #1x256
d['bout'] = np.random.rand(d_out, 1) #1x1

#defining the generator weights:
g = dict()

g['Whl1'] = np.random.rand(g_nhl1, noise_dim) #256x100
g['bhl1'] = np.random.rand(g_nhl1, 1) #256x1

g['Gbn1'] = np.random.rand(g_nhl1, 1) #gamma for BN
g['Bbn1'] = np.random.rand(g_nhl1, 1) #beta for BN

g['Whl2'] = np.random.rand(g_nhl2, g_nhl1) #512x256
g['bhl2'] = np.random.rand(g_nhl2, 1) #512x1

g['Gbn2'] = np.random.rand(g_nhl2, 1) #gamma for BN
g['Bbn2'] = np.random.rand(g_nhl2, 1) #beta for BN

g['Whl3'] = np.random.rand(g_nhl3, g_nhl2) #1024x512
g['bhl3'] = np.random.rand(g_nhl3, 1) #1024x1bn


g['Gbn3'] = np.random.rand(g_nhl3, 1) #gamma for BN
g['Bbn3'] = np.random.rand(g_nhl3, 1) #beta for BN

g['Wout'] = np.random.rand(g_out, g_nhl3) #784x1024
g['bout'] = np.random.rand(g_out, 1) #784x1
g['Gout'] = np.random.rand(g_out, 1) #gamma for BN
g['Bout'] = np.random.rand(g_out, 1) #beta for BN

def activate(matrix, activation = 'leakyReLU'):
	if activation == 'leakyReLU':
		matrix[matrix < 0] *= 0.2
	elif activation == 'ReLU':
		matrix[matrix < 0] = 0
	elif activation == 'sigmoid':
		matrix = 1.0 / (1 + np.exp(-1 * matrix))
	elif activation == 'tanh':
		matrix = np.tanh(matrix)
	return matrix

bnStored = dict()
bnStored['bn1'] = dict()
bnStored['bn1']['means'] = 0
bnStored['bn1']['variances'] = 0

bnStored['bn2'] = dict()
bnStored['bn2']['means'] = 0
bnStored['bn2']['variances'] = 0

bnStored['bn3'] = dict()
bnStored['bn3']['means'] = 0
bnStored['bn3']['variances'] = 0

bnStored['bnout'] = dict()
bnStored['bnout']['means'] = 0
bnStored['bnout']['variances'] = 0
BN_ALPHA = 0.9

def batchNormForwardProp(input, gamma, beta, layerName, train = True):
	#compute means and variances over the minibatch to get normalized output
	global bnStored
	bn = dict()
	if train:
		bn['means'] = np.mean(input, axis = 1)
		bn['means'] = np.reshape(bn['means'], (bn['means'].shape[0] , 1))
		bn['variances'] = np.var(input, axis = 1)
		bn['variances'] = np.reshape(bn['variances'], (bn['variances'].shape[0], 1))
		
		#update the moving averages that are to be used at inference time!
		bnStored[layerName]['means'] = bnStored[layerName]['means'] * BN_ALPHA + (1 - BN_ALPHA) * bn['means']
		bnStored[layerName]['variances'] = bnStored[layerName]['variances'] * BN_ALPHA + (1 - BN_ALPHA) * bn['variances']	

		bn['input_normalized'] = (input - bn['means'])/(bn['variances'] + BN_epsilon)
		# print "Normalized input shape = ",input_normalized.shape
		bn['output'] = bn['input_normalized'] * gamma + beta #n_featuresxbatch_size * n_featuresx1 + nfeaturesx1
		# print "Output shape = ",output.shape

	else:
		bn['input_normalized'] = (input - bnStored[layerName]['means'])/(bnStored[layerName]['variances'] + BN_epsilon)
		# print "Normalized input shape = ",input_normalized.shape
		bn['output'] = bn['input_normalized'] * gamma + beta #n_featuresxbatch_size * n_featuresx1 + nfeaturesx1
		# print "Output shape = ",output.shape'

	return bn

def getGeneratorOutput(nOutputs = 1, train = False):
	#forward prop on the generator nOutput no. of times!
	global g
	act = dict()
	act['noise_vector'] = np.random.normal(0, 1, (noise_dim, nOutputs)) * 1e-4
	
	act['z_hl1'] = np.dot(g['Whl1'], act['noise_vector']) + g['bhl1'] #256x100 * 100xnOut + 256x1
	act['a_hl1'] = activate(act['z_hl1']) #leaky ReLU
	act['bn_hl1'] = batchNormForwardProp(act['a_hl1'], g['Gbn1'], g['Bbn1'], 'bn1', train)

	act['z_hl2'] = np.dot(g['Whl2'], act['bn_hl1']['output']) + g['bhl2'] #512x256 * 256xnOut + 512x1
	act['a_hl2'] = activate(act['z_hl2'])
	act['bn_hl2'] = batchNormForwardProp(act['a_hl2'], g['Gbn2'], g['Bbn2'], 'bn2', train)

	act['z_hl3'] = np.dot(g['Whl3'], act['bn_hl2']['output']) + g['bhl3'] #1024x512 * 512xnOut + 1024x1
	act['a_hl3'] = activate(act['z_hl3'])
	act['bn_hl3'] = batchNormForwardProp(act['a_hl3'], g['Gbn3'], g['Bbn3'], 'bn3', train)

	act['z_out'] = np.dot(g['Wout'], act['bn_hl3']['output']) + g['bout'] #784x1024 * 1024xnOut + 784x1
	
	######EXTRA BN ADDED TO PREVENT SATURATION OF TANH#####
	act['bn_out'] = batchNormForwardProp(act['z_out'], g['Gout'], g['Bout'], 'bnout', train)
	#######################################################

	act['a_out'] = activate(act['bn_out']['output'], activation = 'tanh')
	#acts lie between -1 and 1
	return act #outputs

def getDiscriminatorOutput(input_batch):
	#forward prop on the discriminator to get the output
	global d
	act = dict()
	act['input'] = input_batch
	act['z_hl1'] = np.dot(d['Whl1'], act['input']) + d['bhl1'] #512x784 . 784xbatch_size + 512x1
	act['a_hl1'] = activate(act['z_hl1'])

	act['z_hl2'] = np.dot(d['Whl2'], act['a_hl1']) + d['bhl2'] #256x512 . 512xbatch_size + 256x1
	act['a_hl2'] = activate(act['z_hl2'])

	act['z_out'] = np.dot(d['Wout'], act['a_hl2']) + d['bout'] #1x256 . 256xbatch_size + 1x1
	act['a_out'] = activate(act['z_out'], activation = 'sigmoid')

	return act

def getBatchForTrainingDiscriminator():
	#train discriminator weights on samples from both the generator(-ves) and the dataset(+ves)
	global d, g, allData, batch_size
	#get generator outputs and label them as negative
	#get samples from dataset and label as positive
	#forward prop to get the predictions from disc.
	positives = allData[:, np.random.randint(allData.shape[1], size = batch_size)]
	negatives = getGeneratorOutput(nOutputs = batch_size)['a_out']
	pos_labels = np.ones(batch_size)
	neg_labels = np.zeros(batch_size)
	train_batch = np.hstack([positives, negatives])
	train_batch_labels = np.hstack([pos_labels, neg_labels])
	shuffle_order = np.random.shuffle(np.arange(train_batch.shape[1]))
	train_batch = train_batch[shuffle_order][0]
	train_batch_labels = train_batch_labels[shuffle_order][0]
	train_batch_labels = np.reshape(train_batch_labels, (1,train_batch_labels.shape[0]))
	return train_batch, train_batch_labels
	
def trainDiscriminatorOverBatch(train_batch, train_batch_labels, freezeWeights = True):
	discOut = getDiscriminatorOutput(train_batch)
	preds = discOut['a_out'].astype(np.float64)
	# print preds.shape
	# print train_batch_labels.shape

	#calculate gradient of log error and backprop on it.
	
	dGrad = dict() #storing the gradients wrt the discriminator units
	error = np.average(-1 * train_batch_labels * np.log(preds + smallValue) - (1 - train_batch_labels) * np.log(1- preds + smallValue))
	

	dGrad['a_out'] = np.sum(discOut['a_out'] - train_batch_labels) / train_batch_labels.shape[1]
	dGrad['z_out'] = discOut['a_out'] * (1 - discOut['a_out']) * dGrad['a_out'] #1xb * 1xb one to one, then bcast the output's grad
	dGrad['W_out'] = np.dot(dGrad['z_out'], discOut['a_hl2'].T) #1xb . b x 256
	dGrad['b_out'] = np.sum(dGrad['z_out'], axis = 1) #sum over batch
	dGrad['b_out'] = np.reshape(dGrad['b_out'],(dGrad['b_out'].shape[0], 1))
	dGrad['a_hl2'] = np.dot(d['Wout'].T,dGrad['z_out']) # 256x1 . 1xb = 256xb
	dLeakyRelu = np.ones_like(dGrad['a_hl2'])
	dLeakyRelu[dGrad['a_hl2'] < 0] = 0.2
	
	dGrad['z_hl2'] = dGrad['a_hl2'] * dLeakyRelu #256 x b
	dGrad['W_hl2'] = np.dot(dGrad['z_hl2'], discOut['a_hl1'].T) #256xb . bx512 = 256x512
	dGrad['b_hl2'] = np.sum(dGrad['z_hl2'], axis = 1) #sum over batch
	dGrad['b_hl2'] = np.reshape(dGrad['b_hl2'],(dGrad['b_hl2'].shape[0], 1))
	dGrad['a_hl1'] = np.dot(d['Whl2'].T, dGrad['z_hl2']) # 512x256 . 256xb
	dLeakyRelu = np.ones_like(dGrad['a_hl1'])
	dLeakyRelu[dGrad['a_hl1'] < 0] = 0.2
	
	dGrad['z_hl1'] = dGrad['a_hl1'] * dLeakyRelu #one to one
	dGrad['W_hl1'] = np.dot(dGrad['z_hl1'], discOut['input'].T)#512xb . bx784
	dGrad['b_hl1'] = np.sum(dGrad['z_hl1'], axis = 1)
	dGrad['b_hl1'] = np.reshape(dGrad['b_hl1'],(dGrad['b_hl1'].shape[0], 1))
	dGrad['input'] = np.dot(d['Whl1'].T, dGrad['z_hl1'])#784x512 . 512xb = 784xb
	if not freezeWeights:
		d['Whl1'] -= LR * dGrad['W_hl1']#512x784
		d['bhl1'] -= LR * dGrad['b_hl1']#512x1

		d['Whl2'] -= LR * dGrad['W_hl2']#256x512
		d['bhl2'] -= LR * dGrad['b_hl2']#256x1

		d['Wout'] -= LR * dGrad['W_out']#1x256
		d['bout'] -= LR * dGrad['b_out']#1x1
		
		print 'Weight update of discriminator done!'
	return discOut, dGrad, error

def trainGeneratorOverBatch(freezeWeights = False):
	#train the generator by using both models end to end and freezing weights of discriminator
	global d,g, batch_size
	#generate a batch using gen., assign true labels as positive
	#forward prop on disc. using the batch
	#train gen.'s weights on error in disc.'s predictions

	train_activations = getGeneratorOutput(batch_size, train = True)
	train_batch = train_activations['a_out']
	train_batch_labels = np.ones((1,batch_size))
	
	#dictionary of all of the disc.'s activations, and dictionary of its error gradients
	discOut, gradFromDiscriminator, errorDisc = trainDiscriminatorOverBatch(train_batch, train_batch_labels, freezeWeights = True)
	preds = discOut['a_out']

	error = np.average(-1 * np.log(preds + smallValue)) #since labels are all ones, no need for the zero-label term!

	#propagate the gradient back from the discriminator inputs(=generator outputs), to the generator inputs
	gGrad = dict()
	gGrad['a_out'] = gradFromDiscriminator['input']
	gGrad['bn_out'] = (1 - train_activations['a_out']**2) * gGrad['a_out'] #one to one

	gGrad['Gbn_out'] = np.sum(gGrad['bn_out'] * train_activations['bn_out']['input_normalized'], axis = 1)
	gGrad['Gbn_out'] = np.reshape(gGrad['Gbn_out'], (gGrad['Gbn_out'].shape[0], 1))
	
	gGrad['Bbn_out'] = np.sum(gGrad['bn_out'], axis = 1)
	gGrad['Bbn_out'] = np.reshape(gGrad['Bbn_out'], (gGrad['Bbn_out'].shape[0], 1))

	gGrad['z_out_norm'] = gGrad['bn_out'] * g['Gout'] 
	gGrad['z_out_var'] = np.sum(gGrad['z_out_norm'] * (train_activations['z_out'] - train_activations['bn_out']['means']), axis = 1)
	std_inv = 1./ np.sqrt(train_activations['bn_out']['variances'] + BN_epsilon)
	gGrad['z_out_mean'] = np.sum(gGrad['z_out_norm'] * (-1. * std_inv), axis = 1) + gGrad['z_out_var'] * np.mean(-2. * train_activations['bn_out']['input_normalized'], axis = 1)
	gGrad['z_out_var'] = np.reshape(gGrad['z_out_var'], (gGrad['z_out_var'].shape[0],1))
	gGrad['z_out_mean'] = np.reshape(gGrad['z_out_mean'], (gGrad['z_out_mean'].shape[0],1))
	# print 'grad @ bn out : ', gGrad['bn_out'].shape
	# print 'grad @ z out norm : ', gGrad['z_out_norm'].shape
	# print 'grad @ z out mean : ', gGrad['z_out_mean'].shape
	# print 'grad @ z out var : ', gGrad['z_out_var'].shape

	gGrad['z_out'] = (gGrad['z_out_norm'] * std_inv) + (gGrad['z_out_var'] * 2 * train_activations['bn_out']['input_normalized'] / batch_size) + (gGrad['z_out_mean'] / batch_size) #784xb
	gGrad['W_out'] = np.dot(gGrad['z_out'],train_activations['bn_hl3']['output'].T) #784xb.bx1024
	gGrad['b_out'] = np.sum(gGrad['z_out'], axis = 1)
	gGrad['b_out'] = np.reshape(gGrad['b_out'],(gGrad['b_out'].shape[0],1))
	gGrad['bn_hl3'] = np.dot(g['Wout'].T, gGrad['z_out']) #1024x784 . 784xb

	gGrad['Gbn_hl3'] = np.sum(gGrad['bn_hl3'] * train_activations['bn_hl3']['input_normalized'], axis = 1)
	gGrad['Gbn_hl3'] = np.reshape(gGrad['Gbn_hl3'], (gGrad['Gbn_hl3'].shape[0], 1))
	
	gGrad['Bbn_hl3'] = np.sum(gGrad['bn_hl3'], axis = 1)
	gGrad['Bbn_hl3'] = np.reshape(gGrad['Bbn_hl3'], (gGrad['Bbn_hl3'].shape[0], 1))

	gGrad['a_hl3_norm'] = gGrad['bn_hl3'] * g['Gbn3'] 
	gGrad['a_hl3_var'] = np.sum(gGrad['a_hl3_norm'] * (train_activations['a_hl3'] - train_activations['bn_hl3']['means']), axis = 1)
	std_inv = 1./ np.sqrt(train_activations['bn_hl3']['variances'] + BN_epsilon)
	gGrad['a_hl3_mean'] = np.sum(gGrad['a_hl3_norm'] * (-1. * std_inv), axis = 1) + gGrad['a_hl3_var'] * np.mean(-2. * train_activations['bn_hl3']['input_normalized'], axis = 1)
	gGrad['a_hl3_var'] = np.reshape(gGrad['a_hl3_var'], (gGrad['a_hl3_var'].shape[0],1))
	gGrad['a_hl3_mean'] = np.reshape(gGrad['a_hl3_mean'], (gGrad['a_hl3_mean'].shape[0],1))
	gGrad['a_hl3'] = (gGrad['a_hl3_norm'] * std_inv) + (gGrad['a_hl3_var'] * 2 * train_activations['bn_hl3']['input_normalized'] / batch_size) + (gGrad['a_hl3_mean'] / batch_size) #784xb

	# print 'grad @ bn hl3 : ', gGrad['bn_hl3'].shape
	# print 'grad @ a hl3 norm : ', gGrad['a_hl3_norm'].shape
	# print 'grad @ a hl3 mean : ', gGrad['a_hl3_mean'].shape
	# print 'grad @ a hl3 var : ', gGrad['a_hl3_var'].shape

	dLeakyRelu = np.ones_like(gGrad['a_hl3'])
	dLeakyRelu[gGrad['a_hl3'] < 0] = 0.2
	gGrad['z_hl3'] = gGrad['a_hl3'] * dLeakyRelu #1024xb
	gGrad['W_hl3'] = np.dot(gGrad['z_hl3'], train_activations['a_hl2'].T) #1024xb . bx512
	gGrad['b_hl3'] = np.sum(gGrad['z_hl3'], axis = 1)
	gGrad['b_hl3'] = np.reshape(gGrad['b_hl3'],(gGrad['b_hl3'].shape[0],1))
	gGrad['bn_hl2'] = np.dot(g['Whl3'].T, gGrad['z_hl3'])# 512x1024 . 1024xb

	gGrad['Gbn_hl2'] = np.sum(gGrad['bn_hl2'] * train_activations['bn_hl2']['input_normalized'], axis = 1)
	gGrad['Gbn_hl2'] = np.reshape(gGrad['Gbn_hl2'], (gGrad['Gbn_hl2'].shape[0], 1))
	
	gGrad['Bbn_hl2'] = np.sum(gGrad['bn_hl2'], axis = 1)
	gGrad['Bbn_hl2'] = np.reshape(gGrad['Bbn_hl2'], (gGrad['Bbn_hl2'].shape[0], 1))

	gGrad['a_hl2_norm'] = gGrad['bn_hl2'] * g['Gbn2'] 
	gGrad['a_hl2_var'] = np.sum(gGrad['a_hl2_norm'] * (train_activations['a_hl2'] - train_activations['bn_hl2']['means']), axis = 1)
	std_inv = 1./ np.sqrt(train_activations['bn_hl2']['variances'] + BN_epsilon)
	gGrad['a_hl2_mean'] = np.sum(gGrad['a_hl2_norm'] * (-1. * std_inv), axis = 1) + gGrad['a_hl2_var'] * np.mean(-2. * train_activations['bn_hl2']['input_normalized'], axis = 1)
	gGrad['a_hl2_var'] = np.reshape(gGrad['a_hl2_var'], (gGrad['a_hl2_var'].shape[0],1))
	gGrad['a_hl2_mean'] = np.reshape(gGrad['a_hl2_mean'], (gGrad['a_hl2_mean'].shape[0],1))
	gGrad['a_hl2'] = (gGrad['a_hl2_norm'] * std_inv) + (gGrad['a_hl2_var'] * 2 * train_activations['bn_hl2']['input_normalized'] / batch_size) + (gGrad['a_hl2_mean'] / batch_size) #784xb

	# print 'grad @ bn hl2 : ', gGrad['bn_hl2'].shape
	# print 'grad @ a hl2 norm : ', gGrad['a_hl2_norm'].shape
	# print 'grad @ a hl2 mean : ', gGrad['a_hl2_mean'].shape
	# print 'grad @ a hl2 var : ', gGrad['a_hl2_var'].shape


	dLeakyRelu = np.ones_like(gGrad['a_hl2'])
	dLeakyRelu[gGrad['a_hl2'] < 0] = 0.2
	gGrad['z_hl2'] = gGrad['a_hl2'] * dLeakyRelu #1024xb
	gGrad['W_hl2'] = np.dot(gGrad['z_hl2'], train_activations['a_hl1'].T) #1024xb . bx512
	gGrad['b_hl2'] = np.sum(gGrad['z_hl2'], axis = 1)
	gGrad['b_hl2'] = np.reshape(gGrad['b_hl2'],(gGrad['b_hl2'].shape[0],1))
	gGrad['bn_hl1'] = np.dot(g['Whl2'].T, gGrad['z_hl2'])# 512x1024 . 1024xb

	gGrad['Gbn_hl1'] = np.sum(gGrad['bn_hl1'] * train_activations['bn_hl1']['input_normalized'], axis = 1)
	gGrad['Gbn_hl1'] = np.reshape(gGrad['Gbn_hl1'], (gGrad['Gbn_hl1'].shape[0], 1))
	
	gGrad['Bbn_hl1'] = np.sum(gGrad['bn_hl1'], axis = 1)
	gGrad['Bbn_hl1'] = np.reshape(gGrad['Bbn_hl1'], (gGrad['Bbn_hl1'].shape[0], 1))

	gGrad['a_hl1_norm'] = gGrad['bn_hl1'] * g['Gbn1'] 
	gGrad['a_hl1_var'] = np.sum(gGrad['a_hl1_norm'] * (train_activations['a_hl1'] - train_activations['bn_hl1']['means']), axis = 1)
	std_inv = 1./ np.sqrt(train_activations['bn_hl1']['variances'] + BN_epsilon)
	gGrad['a_hl1_mean'] = np.sum(gGrad['a_hl1_norm'] * (-1. * std_inv), axis = 1) + gGrad['a_hl1_var'] * np.mean(-2. * train_activations['bn_hl1']['input_normalized'], axis = 1)
	gGrad['a_hl1_var'] = np.reshape(gGrad['a_hl1_var'], (gGrad['a_hl1_var'].shape[0],1))
	gGrad['a_hl1_mean'] = np.reshape(gGrad['a_hl1_mean'], (gGrad['a_hl1_mean'].shape[0],1))
	gGrad['a_hl1'] = (gGrad['a_hl1_norm'] * std_inv) + (gGrad['a_hl1_var'] * 2 * train_activations['bn_hl1']['input_normalized'] / batch_size) + (gGrad['a_hl1_mean'] / batch_size) #784xb

	# print 'grad @ bn hl1 : ', gGrad['bn_hl2'].shape
	# print 'grad @ a hl1 norm : ', gGrad['a_hl1_norm'].shape
	# print 'grad @ a hl1 mean : ', gGrad['a_hl1_mean'].shape
	# print 'grad @ a hl1 var : ', gGrad['a_hl1_var'].shape

	dLeakyRelu = np.ones_like(gGrad['a_hl1'])
	dLeakyRelu[gGrad['a_hl1'] < 0] = 0.2
	gGrad['z_hl1'] = gGrad['a_hl1'] * dLeakyRelu #1024xb
	gGrad['W_hl1'] = np.dot(gGrad['z_hl1'], train_activations['noise_vector'].T) #1024xb . bx512
	gGrad['b_hl1'] = np.sum(gGrad['z_hl1'], axis = 1)
	gGrad['b_hl1'] = np.reshape(gGrad['b_hl1'],(gGrad['b_hl1'].shape[0],1))
	if not freezeWeights:
		g['Whl1'] -= LR * gGrad['W_hl1']
		g['bhl1'] -= LR * gGrad['b_hl1']

		g['Gbn1'] -= LR * gGrad['Gbn_hl1']
		g['Bbn1'] -= LR * gGrad['Bbn_hl1']

		g['Whl2'] -= LR * gGrad['W_hl2']
		g['bhl2'] -= LR * gGrad['b_hl2']

		g['Gbn2'] -= LR * gGrad['Gbn_hl2']
		g['Bbn2'] -= LR * gGrad['Bbn_hl2']

		g['Whl3'] -= LR * gGrad['W_hl3']
		g['bhl3'] -= LR * gGrad['b_hl3']

		g['Gbn3'] -= LR * gGrad['Gbn_hl3']
		g['Bbn3'] -= LR * gGrad['Bbn_hl3']

		g['Wout'] -= LR * gGrad['W_out']
		g['bout'] -= LR * gGrad['b_out']

		g['Gout'] -= LR * gGrad['Gbn_out']
		g['Bout'] -= LR * gGrad['Bbn_out']

	return train_activations, gGrad, error

#training loop:
print getDiscriminatorOutput(allData[:, 0])['a_hl1']
exit()
for i in range(100):
	print '###################EPOCH :', (i + 1)
	for j in range(50):
		print 'Training the Discriminator : Iteration :', (j + 1)
		training_data, training_labels = getBatchForTrainingDiscriminator()
		discOut, dGrad, discError = trainDiscriminatorOverBatch(training_data, training_labels, freezeWeights = False)
		if(j % 10 == 0):
			# print 'Discriminator Error = ', discError
			print dGrad['input']
	for k in range(50):
		print 'Training the Generator : Iteration :', (k + 1)
		genOut, gGrad, genError = trainGeneratorOverBatch(freezeWeights = False)
		if(k % 10 == 0):
			print 'Generator Error = ', genError
			print getGeneratorOutput()['a_out']
		exit()
