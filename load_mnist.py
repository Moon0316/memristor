#_*_coding:utf-8_*_
import numpy as np
import os
class Mnist(object):

    def __init__(self):

        self.dataname = "Mnist"
        self.dims = 28*28
        self.shape = [28 , 28 , 1]
        self.image_size = 28


    def load_mnist(self,type):

        data_dir = "./data"
        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd , dtype=np.uint8)
		
        trX = loaded[16:].reshape((60000, 28 , 28 ,  1)).astype(np.float64)
        point = loaded[:16]
        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float64)
		
        point = loaded[:8]

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28 , 28 , 1)).astype(np.float64)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float64)

        trY = np.asarray(trY)
        teY = np.asarray(teY)


        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)

        X = X.reshape(70000,784)
		
        y_vec = np.zeros((len(y), 10), dtype=np.float64)
        for i, label in enumerate(y):
            y_vec[i, int(y[i])] = 1.0

        if type == "train":
            return X[:60000],y_vec[:60000]
        if type == "test":
            return X[60000:],y_vec[60000:]
        if type == "all":
            return X,y_vec


def process_data(type):
    mn_dataset = Mnist()
    return mn_dataset.load_mnist(type)