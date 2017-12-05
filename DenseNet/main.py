import tensorflow as tf
import numpy as np
import os
import time
import argparse
import densenet
from densenet import Model #Class Name

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="Set the learning rate (Default = 0.001)",
                    type=float, default = 0.001)
parser.add_argument("--batch_size", help="Set the batch size (Default = 20)",
                    type=int, default = 20)
parser.add_argument("--growth_rate", help="Growth Rate, k (Default = 12) ",
                    type=int, default = 12)
parser.add_argument("--decay", help="Batchnorm decay (Default = 0.99)",
                    type=float, default = 0.99)
parser.add_argument("--epochs", help = "Epochs (Default = 200)",
					type = int, default = 200 )
parser.add_argument("--reduction", help = "Reduction, theta (Default = 0.5)",
					type = float, default = 0.5 )
parser.add_argument("--blocks", help = "Dense Blocks Count (Default = 3)",	
					type = int, default = 3)
parser.add_argument("--depth", help = "Depth, L (Default = 100)",
					type = int, default = 100)
parser.add_argument("--augment", help = "Enable Data Augmentation (Default = True)",	
					type = bool, default = True)
parser.add_argument("--val_fraction", help = "Fraction of dataset to be split for validation (Default = 0.2)",
					type = float, default = 0.2)
parser.add_argument("--bottleneck", help = "Enable Bottleneck (Default = True)",
					type = bool, default = True)
parser.add_argument("--ckpt_dir", help = "Set checkpoint_dir",
					default = './checkpoint')
parser.add_argument("--logdir", help = "Set tensorboard logdir",
					default = './tensorboard')
parser.add_argument("--restore", help = "Restore Checkpoint for Training (Default = False)",
					type = bool, default = False)
parser.add_argument("--mode", help = "Train/Test")


args = parser.parse_args()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

net = Model(args.epochs, args.augment, args.lr, args.decay, args.growth_rate, args.depth, args.reduction,
			args.bottleneck, args.blocks, args.val_fraction, args.ckpt_dir, args.logdir, args.batch_size, args.restore)

if args.mode == "Train":

	with open("args.txt", "w") as file:
		for arg in vars(args):
			text = arg + " = " + str(getattr(args,arg)) + "\n"
			file.write(text)

	net.Train()

if args.mode == "Test":

	#Note: Use the same arguments saved in args.txt before testing
	net.Test()
