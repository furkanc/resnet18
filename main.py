import tensorflow as tf
import utils
import resnet_model
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-path', default='Images/', dest='path', help='Path for images')
parser.add_argument('-size', default=224, dest='size',type=int, help='Fixed image size')
parser.add_argument('-batch_size', dest='batch_size', type=int, help='Batch Size')
parser.add_argument('-logdir', default='log', dest='logdir', help='Log directory for Tensorflow')
parser.add_argument('-lr', default=1e-1, dest='lr', type =np.float32, help='Learning Rate')
args = parser.parse_args()

logdir = args.logdir
path = args.path


SIZE = args.size
CLASSES = 120
mean = np.array([121.41, 111.21, 99.71])
mean = mean.reshape((1, 1, 3))
LR = args.lr 

train_data, train_label, test_data, test_label = utils.get_data(path)
print(len(train_data))
model = resnet_model.Model()
model.build(SIZE, CLASSES, logdir, LR)
model.run(train_data, train_label, args.batch_size, CLASSES, SIZE, mean)

