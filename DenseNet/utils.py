import tensorflow as tf
import numpy as np
import os
import pickle
import skimage as S
from skimage import transform

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data(val_fraction):

	train_fraction = 1-val_fraction
	n_train = int(train_fraction*50000)
	data = []
	label = []

	for i in range(5):
	    x = []
	    try:
	        x = unpickle('data_batch_' + str(i+1))
	    except:
	        print('Place all "data_batch" files (1 to 5) in the ip directory')
	        print('Download From: https://www.cs.toronto.edu/~kriz/cifar.html')
	        print('...Terminating...')
	        exit()
	    images = x[b'data']
	    labval = x[b'labels']
	    R = images[:,0:1024]
	    G = images[:,1024:2048]
	    B = images[:,2048:3072]
	    R = np.reshape(R,(-1,32,32,1))
	    G = np.reshape(G,(-1,32,32,1))
	    B = np.reshape(B,(-1,32,32,1))
	    temp = np.concatenate((R,G,B),axis = 3)
	    if i == 0:
	        data = temp
	        label = labval
	    else:
	        data = np.concatenate((data,temp), axis = 0)
	        label = np.concatenate((label,labval), axis = 0)

	P = np.random.permutation(50000)
	data2 = data[P]
	label2 = label[P]

	data_train = data2[0 : n_train]
	label_train = label2[0 : n_train]
	data_val = data2[n_train : ]
	label_val = label2[n_train : ]

	return data_train, label_train, data_val, label_val

def get_data_test():

    try:
        x = unpickle('test_batch')
    except:
        print('Place test batch file in the ip directory')
        print('Download From: https://www.cs.toronto.edu/~kriz/cifar.html')
        print('...Terminating...')
        exit()
    images = x[b'data']
    labval = x[b'labels']
    R = images[:,0:1024]
    G = images[:,1024:2048]
    B = images[:,2048:3072]
    R = np.reshape(R,(-1,32,32,1))
    G = np.reshape(G,(-1,32,32,1))
    B = np.reshape(B,(-1,32,32,1))
    temp = np.concatenate((R,G,B),axis = 3)
    data = temp
    label = labval

    return data, label


def preprocess(images, label, augment, val_fraction):

	t = 1 - val_fraction
	perm = np.random.permutation(int(50000*t))
	shuffled = images[perm]
	lab = label[perm]

	if not augment:
		return shuffled, lab

	elif augment:
		p = int(50000*t) // 10 #20% flipped, 40% cropped, 40% normal data

		part1 = shuffled[:p*2]
		part2 = (shuffled[p*2:p*3])[:,8:,8:,:]
		part3 = (shuffled[p*3:p*4])[:,8:,:24,:]
		part4 = (shuffled[p*4:p*5])[:,:24,8:,:]
		part5 = (shuffled[p*5:p*6])[:,:24,:24,:]
		t6 = shuffled[p*6:]

		t1 = []
		t2 = []
		t3 = []
		t4 = []
		t5 = []

		mode = 'constant'

		for i in range(2*p):
			t1.append(np.fliplr(part1[i]))
			if(i<p):
				t2.append(S.transform.resize(part2[i],(32,32,3), mode = mode))
				t3.append(S.transform.resize(part3[i],(32,32,3), mode = mode))
				t4.append(S.transform.resize(part4[i],(32,32,3), mode = mode))
				t5.append(S.transform.resize(part5[i],(32,32,3), mode = mode))

		return np.concatenate((t1,t2,t3,t4,t5,t6), axis = 0), lab

def normalize(ip): #Not used, channel wise norm
	
	for i in range(3):
		ip[:,:,:,i] = (ip[:,:,:,i] - np.mean(ip[:,:,:,i]))/np.std(ip[:,:,:,i])
	return ip 

def unnormalize(ip): #Not used
	
	return (((ip + 1)/2) * 255).astype(np.uint8) #[-1,1] -> [0,255]

def Conv(ip, filter, stride, output_ch):
	input_ch = ip.get_shape()[3]
	h = tf.get_variable("Filter", shape = [filter, filter, input_ch, output_ch], dtype = tf.float32, initializer = tf.keras.initializers.he_normal())
	b = tf.get_variable("Bias", shape = [output_ch], dtype = tf.float32, initializer = tf.constant_initializer(0))
	return tf.nn.bias_add(tf.nn.conv2d(ip, h, strides = [1, stride, stride, 1], padding = "SAME"), b)

def MaxPool(ip):
	return tf.nn.max_pool(ip, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def AvgPool(ip, k=2):
	return tf.nn.avg_pool(ip, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def Relu(ip):
	return tf.nn.relu(ip)

def BatchNorm(ip, isTrain, decay = 0.99):
	return tf.contrib.layers.batch_norm(ip, is_training = isTrain, decay = decay)