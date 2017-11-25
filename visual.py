import matplotlib.pyplot as plt 
import numpy as np 

def show_curves():
	filterr = np.load('filters.npy')
	f, axarr = plt.subplots(8,8)
	for i in range(8):
	    for j in range(8):
	        axarr[i,j].imshow(filterr[:,:,:,(i*8)+j])
	        axarr[i,j].axis('off')
	plt.show()

if __name__ == "__main__":
	show_curves() 