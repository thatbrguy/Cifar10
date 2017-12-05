import tensorflow as tf
import numpy as np
import os
import time
import utils
from utils import *

class Model():

	def __init__(self, epochs, augment, lr, decay, growth_rate, depth,
	 			reduction, bottleneck, block, val_fraction, ckpt_dir,
	 			logdir, batch_sz, restore):

		self.epochs = epochs
		self.augment = augment
		self.lr = lr
		self.decay = decay
		self.growth_rate = growth_rate #k in paper
		self.depth = depth #L in paper
		self.reduction = reduction #Theta in paper
		self.BottleNeck_Enable = bottleneck
		self.block_count = block
		self.layers_per_block = (self.depth - (self.block_count + 1)) // self.block_count
		self.val_fraction = val_fraction
		self.ckpt_dir = ckpt_dir
		self.logdir = logdir
		self.batch_sz = batch_sz
		self.restore = restore

		if self.BottleNeck_Enable:
			self.layers_per_block = self.layers_per_block // 2 

	def Layer(self, ip):

		with tf.variable_scope("Composite"):
			next_layer = BatchNorm(ip, self.isTrain, self.decay)
			next_layer = Relu(next_layer)
			next_layer = Conv(next_layer, filter = 3, stride = 1, output_ch = self.growth_rate)

			return next_layer

	def Transition(self, ip, name):

		with tf.variable_scope(name):
			reduced_output_size  = int(int(ip.get_shape()[-1]) * self.reduction)
			next_layer = BatchNorm(ip, self.isTrain, self.decay)
			next_layer = Conv(next_layer, filter = 3, stride =1, output_ch = reduced_output_size)
			next_layer = AvgPool(next_layer)

			return next_layer

	def BottleNeck(self, ip):

		with tf.variable_scope("BottleNeck"):
			next_layer = BatchNorm(ip, self.isTrain, self.decay)
			next_layer = Relu(next_layer)
			next_layer = Conv(next_layer, filter = 1, stride = 1, output_ch = self.growth_rate*4)

			return next_layer

	def Block(self, ip, name):
		
		if self.BottleNeck_Enable:
			output = self.BottleNeck(ip)
			output = self.Layer(output)
		elif not self.BottleNeck_Enable:
			output = self.Layer(ip)

		return tf.concat([ip, output], axis = 3)

	def BuildBlock(self, ip, name):

		with tf.variable_scope(name):
			for i in range(self.layers_per_block):
				with tf.variable_scope("Layer" + str(i+1)) as scope:
					output = self.Block(ip, scope)
					ip = output

		return output

	def BuildModel(self):

		with tf.name_scope('Placeholders'):
			self.X = tf.placeholder(name = 'Input', shape = [None, 32, 32, 3], dtype = tf.float32)
			self.Lab = tf.placeholder(name = 'Label', shape = None, dtype = tf.int32)
			self.isTrain = tf.placeholder(name = 'isTrain', shape = None, dtype = tf.bool)
		
		with tf.variable_scope('Input'):
			Input = Conv(self.X, filter = 3, stride = 1, output_ch = self.growth_rate * 2)
			
		for i in range(self.block_count):
			Input = self.BuildBlock(Input, "Block" + str(i+1))
			if(i != self.block_count-1):
				Input = self.Transition(Input, "Transition" + str(i+1))

		with tf.variable_scope("Classification"):
			Input = BatchNorm(Input, self.isTrain, self.decay)
			Input = Relu(Input)
			final = AvgPool(Input, k = Input.get_shape()[-2])
			#shape =  final.get_shape()
			#sh = shape[1]*shape[2]*shape[3]
			f = tf.reshape(final, [-1, final.get_shape()[-1]])
			W = tf.get_variable("FC", shape = [f.get_shape()[-1], 10], initializer = tf.keras.initializers.he_normal())
			b = tf.get_variable("FC_Bias", shape = [10], initializer = tf.constant_initializer(0))
			self.logits = tf.matmul(f, W) + b

		with tf.variable_scope("Loss"):
			self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.Lab,logits = self.logits))
			self.cost_sum = tf.summary.scalar('Cost', self.cost)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
			self.saver = tf.train.Saver(max_to_keep = 2)


	def Train(self):
		
		print('Loading Data')

		data_train, label_train, data_val, label_val = get_data(self.val_fraction)

		print('Data Loaded')

		data_train = (data_train - np.mean(data_train)) / np.std(data_train)
		data_val = (data_val - np.mean(data_val)) / np.std(data_val)

		train_data_len = int((1- self.val_fraction) * 50000)
		val_data_len = int((self.val_fraction) * 50000)

		epochs = self.epochs
		batch_sz = self.batch_sz
		batches = train_data_len // batch_sz
		val_batches = val_data_len // batch_sz
		acc_train_epochs = []
		acc_val_epochs = []
		total_cost = []
		val_best = -1

		print('Loading Model')
		self.BuildModel()
		print('Model Loaded')
		
		parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
		self.writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())

		with tf.Session() as sess:

			init_op = tf.global_variables_initializer()
			sess.run(init_op)
			step = 0
			
			if self.restore:
				print('Restoring Checkpoint')
				ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
				self.saver.restore(sess, ckpt)
				print('Checkpoint Restored')

			print('Paramters:', sess.run(parameter_count))


			for i in range(epochs):
				correct = 0
				p = [] 
				c = [] 
				p_val = [] 

				shuffle_data, shuffle_label = preprocess(data_train, label_train, self.augment, self.val_fraction)

				for j in range(batches):

					begin = time.time()
					batch = np.arange(j*batch_sz, (j+1)*batch_sz)
					B_Data = shuffle_data[batch]
					B_Label = shuffle_label[batch]

					sess.run(self.optimizer, feed_dict = {self.X: B_Data, self.Lab: B_Label, self.isTrain: True})

					c.append(sess.run(self.cost, feed_dict={self.X: B_Data, self.Lab: B_Label, self.isTrain: True}))

					p.append(self.logits.eval(feed_dict = {self.X: B_Data, self.isTrain: True}, session = sess))

					cw = sess.run(self.cost_sum, feed_dict={self.X: B_Data, self.Lab: B_Label, self.isTrain: True})	

					step = step + 1
					self.writer.add_summary(cw, step)


					if((j+1)%20 == 0):
						print('Batch:',j,'Cost:',c[j],'Time/Batch:', time.time() - begin, end = '\r')
					

				for j in range(val_batches):
					batch = np.arange(j*batch_sz, (j+1)*batch_sz)
					B_Val = data_val[batch]
					p_val.append(self.logits.eval(feed_dict = {self.X: B_Val, self.isTrain: False}, session = sess))
					
			
				temp = np.reshape( p, (train_data_len,10) )
				temp_val = np.reshape( p_val, (val_data_len,10) )
				correct = np.equal( np.argmax( temp, axis=1), shuffle_label )
				correct_val = np.equal( np.argmax( temp_val, axis=1), label_val )
				count = np.sum( correct.astype(np.int32) )
				count_val = np.sum( correct_val.astype(np.int32) )
				acc = count/train_data_len
				acc_val = count_val/val_data_len

				if((i+1)%50==0):
					lr = lr/2

				bcost = np.sum(c)
				acc_train_epochs.append(acc)
				acc_val_epochs.append(acc_val)
				total_cost.append(bcost)
			

				if(i%5==0):
					if(acc_val >  val_best):
						val_best = acc_val
						self.saver.save(sess, self.ckpt_dir + '/densenet', global_step = step)

				print('Epoch:',i,'Train:',acc,'Validation:',acc_val,'Cost:',bcost)

				np.save('test.npy', acc_train_epochs)
				np.save('val.npy', acc_val_epochs)
				np.save('cost.npy', total_cost)

	def Test(self):

		batch_sz = self.batch_sz
		test_data_len = 10000
		p_test = []

		batches = test_data_len // batch_sz

		print('...Loading Test Data...')
		data, label = get_data_test()
		print('...Test Data Loaded...')

		print('...Loading Model...')
		self.BuildModel()
		print('...Model Loaded...')

		with tf.Session() as sess:

			init_op = tf.global_variables_initializer()
			sess.run(init_op)

			print('...Restoring Checkpoint...')
			ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
			self.saver.restore(sess, ckpt)

			print('...Checkpoint Restored...')
			
			data_test = (data - np.mean(data)) / np.std(data)

			for i in range(batches):

				batch = np.arange(i*batch_sz, (i+1)*batch_sz)
				B_Data = data_test[batch]

				p_test.append(self.logits.eval(feed_dict = {self.X: B_Data, self.isTrain: False}, session = sess))

			temp = np.reshape(p_test, (test_data_len,10))
			correct = np.equal(np.argmax(temp, axis=1),label)
			count = np.sum(correct.astype(np.int32))
			acc = count/test_data_len

			print('Test Accuracy: ', acc)
