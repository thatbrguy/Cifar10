import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import skimage as S
from skimage import transform

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="Set the learning rate (Default = 0.0001)",
                    type=int, default=0.0001)
parser.add_argument("--batch_size", help="Set the batch size (Default = 20)",
                    type=int, default=20)
parser.add_argument("--init", help="Xavier: 1, He: 2 (Default = Xavier)",
                    type=int, default=1)
parser.add_argument("--save_dir", help="Save output files to this directory (Default = Current Directory)",
                     default = '')

args = parser.parse_args()
direc = './' + args.save_dir

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data():
	data = []
	label = []

	for i in range(5):
	    x = unpickle('data_batch_' + str(i+1))
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

	data_train = data2[0:45000]
	label_train = label2[0:45000]
	data_val = data2[45000:]
	label_val = label2[45000:]

	return data_train, label_train, data_val, label_val

def init_weight(shape, init_type, name):
    if init_type == 1: #Xavier
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    else: #He_Normal
        return  tf.get_variable(name, shape, initializer= tf.keras.initializers.he_normal())
    
def conv(ip, weight, bias):
    return (tf.nn.conv2d(ip, weight, strides=[1, 1, 1, 1], padding='SAME') + bias)

def maxpool(ip, stride = [1,2,2,1], ksize = [1,2,2,1]):
    return tf.nn.max_pool(ip, ksize=ksize, strides=stride, padding='SAME')

def BN(ip, isTrain, decay):
    return tf.contrib.layers.batch_norm(ip, is_training = isTrain, decay = decay)

def preprocess(images, label):
    
    perm = np.random.permutation(45000)
    shuffled = images[perm]
    lab = label[perm]
    
    p = 45000 // 10 #20% flipped, 40% cropped, 40% normal data
    
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

    for i in range(9000):
        t1.append(np.fliplr(part1[i]))
        if(i<4500):
            t2.append(S.transform.resize(part2[i],(32,32,3)))
            t3.append(S.transform.resize(part3[i],(32,32,3)))
            t4.append(S.transform.resize(part4[i],(32,32,3)))
            t5.append(S.transform.resize(part5[i],(32,32,3)))

    return np.concatenate((t1,t2,t3,t4,t5,t6), axis = 0), lab

def train():

	epochs = 1000
	batch_sz = 20
	batches = 45000 // batch_sz
	val_batches = 5000 // batch_sz
	lr = 0.0001
	acc_epochs = []
	acc_val_epochs = []
	total_cost = []
	val_best = -1

	tf.reset_default_graph()
	init = args.init

	X = tf.placeholder(dtype = tf.float32, name="Input", shape=[None,32,32,3])
	Lab = tf.placeholder(dtype = tf.int32, name = "Label", shape = [None])
	LR = tf.placeholder(dtype = tf.float32, name = 'LR')
	isTrain = tf.placeholder(dtype = tf.bool, name = 'isTrain')
	decay = tf.placeholder(dtype = tf.float32, name="Decay", shape = None)


	with tf.variable_scope('Conv1'):
	    Wconv1 = init_weight([3,3,3,64], init, 'W_Conv1' )
	    bconv1 = init_weight([64], init, 'b_Conv1' )
	    hconv1 = tf.nn.relu( conv(X, Wconv1, bconv1) )
	    BNconv1 =  BN(hconv1, isTrain, decay)
	    
	with tf.variable_scope('Pool1'):
	    hpool1 = maxpool(BNconv1)

	with tf.variable_scope('Conv2'):
	    Wconv2 = init_weight([3,3,64,128], init, 'W_Conv2')
	    bconv2 = init_weight([128], init, 'b_Conv2')
	    hconv2 = tf.nn.relu( conv(hpool1, Wconv2, bconv2) )
	    BNconv2 =  BN(hconv2, isTrain, decay)

	with tf.variable_scope('Pool2'):
	    hpool2 = maxpool(BNconv2)
	    
	with tf.variable_scope('Conv3'):
	    Wconv3 = init_weight([3,3,128,256], init, 'W_Conv3')
	    bconv3 = init_weight([256], init, 'b_Conv3')
	    hconv3 = tf.nn.relu( conv(hpool2, Wconv3, bconv3) )
	    BNconv3 =  BN(hconv3, isTrain, decay)

	with tf.variable_scope('Conv4'):
	    Wconv4 = init_weight([3,3,256,256], init, 'W_Conv4')
	    bconv4 = init_weight([256], init, 'b_Conv4')
	    hconv4 = tf.nn.relu( conv(BNconv3, Wconv4, bconv4) )
	    BNconv4 =  BN(hconv4, isTrain, decay)

	with tf.variable_scope('Pool3'):
	    hpool3 = maxpool(BNconv4)
	    
	shape = hpool3.get_shape().as_list()
	sh = shape[1]*shape[2]*shape[3]

	FC_IP = tf.reshape(hpool3, [-1, sh] )

	with tf.variable_scope('FC1'):
	    Wfc1 = init_weight([sh,1024], init, 'W_FC1')
	    bfc1 = init_weight([1024], init, 'b_FC1')
	    hfc1 = tf.nn.relu(tf.matmul(FC_IP, Wfc1) + bfc1)
	    #hfc1 = tf.nn.tanh(tf.matmul(FC_IP, Wfc1) + bfc1)
	    BNfc1 =  BN(hfc1, isTrain, decay)
	    
	with tf.variable_scope('FC2'):
	    Wfc2 = init_weight([1024,1024],init,'W_FC2')
	    bfc2 = init_weight([1024],init,'b_FC2')
	    hfc2 = tf.nn.relu(tf.matmul(hfc1, Wfc2) + bfc2)
	    #hfc2 = tf.nn.tanh(tf.matmul(hfc1, Wfc2) + bfc2)
	    BNfc2 =  BN(hfc2, isTrain, decay)
	    
	#Creating one more weight matrix to downscale from 1024 to 10 for softmax
	    
	with tf.variable_scope('Softmax'):
	    Wfc3 = init_weight([1024,10],init,'W_softmax')
	    bfc3 = init_weight([10],init,'b_softmax')
	    #bfc3 = tf.Variable(tf.zeros([10]),name = 'b_softmax')
	    hfc3 = (tf.matmul(hfc2, Wfc3) +bfc3)
	    #hfc3 = tf.nn.relu(tf.matmul(hfc2, Wfc3) +bfc3)
	    BNfc3 =  BN(hfc3, isTrain, decay)
	    
	    #hfcBN = bn_layer_top(hfc3, 'bn', isTrain, decay = decay)
	    
	BNfc3.get_shape().as_list()

	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = Lab,logits = BNfc3))

	saver = tf.train.Saver()

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
	    optimizer = tf.train.AdamOptimizer(LR).minimize(cost)

	data_train, label_train, data_val, label_val = get_data()

	#Normalising data
	data_train = (data_train - np.mean(data_train))/np.std(data_train)
	data_val = (data_val - np.mean(data_val))/np.std(data_val)

	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    
	    for i in range(epochs):
	        correct = 0
	        p = []
	        c = []
	        p_val = []
	        
	        shuffle_data, shuffle_label = preprocess(data_train, label_train)
	        
	        for j in range(batches):
	            batch = np.arange(j*batch_sz, (j+1)*batch_sz)
	            
	            B_Data = shuffle_data[batch]
	            B_Label = shuffle_label[batch]
	            
	            sess.run(optimizer, feed_dict = {X: B_Data, Lab: B_Label, LR: lr, isTrain: True, decay: 0.99})
	            c.append(sess.run(cost, feed_dict={X: B_Data, Lab: B_Label, isTrain: True, decay: 0.99}))
	            p.append(BNfc3.eval(feed_dict = {X: B_Data, isTrain: True, decay: 0.99}, session = sess))
	            
	            if((j+1)%200 == 0):
	                print('Batch:',j,'Cost:',c[j], end = '\r')
	        
	        for j in range(val_batches):
	            batch = np.arange(j*batch_sz, (j+1)*batch_sz)
	            B_Val = data_val[batch]
	            p_val.append(BNfc3.eval(feed_dict = {X: B_Val, isTrain: False, decay:0.99}, session = sess))
	         
	        temp = np.reshape( p, (45000,10) )
	        temp_val = np.reshape( p_val, (5000,10) )
	        correct = np.equal( np.argmax( temp, axis=1), shuffle_label )
	        correct_val = np.equal( np.argmax( temp_val, axis=1), label_val )
	        count = np.sum( correct.astype(np.int32) )
	        count_val = np.sum( correct_val.astype(np.int32) )
	        acc = count/45000
	        acc_val = count_val/5000
	        
	        if((i+1)%50==0):
	            lr = lr/2
	        
	        bcost = np.sum(c)
	        acc_epochs.append(acc)
	        acc_val_epochs.append(acc_val)
	        total_cost.append(bcost)
	        
	        if(i%5==0):
	            if(acc_val >  val_best):
	                val_best = acc_val
	                fname = direc + '/cifar_10_EP_' + str(i) + '.ckpt'
	                saver.save(sess, fname)
	                
	        print('Epoch:',i,'Train:',acc,'Validation:',acc_val,'Cost:',bcost)

if __name__ == '__main__':
	train()

