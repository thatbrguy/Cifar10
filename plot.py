import matplotlib.pyplot as plt 
import numpy as np 

def show_curves():

	#Keep the respective npy files in the same directory before executing
	y = 100*np.load('test.npy')
	z = 100*np.load('val.npy')
	x = range(np.shape(y)[0])
	plt.plot(x,y, label='Train')
	#plt.hold(True)
	plt.plot(x,z, label='Val')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.axis([0,100,0,110])
	plt.legend()
	plt.title('Accuracy Plots')
	plt.show()


if __name__ == "__main__":
	show_curves() 