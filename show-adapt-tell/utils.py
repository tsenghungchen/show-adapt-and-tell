import tensorflow as tf
import time
import json
import h5py
from functools import reduce
from tensorflow.contrib.layers.python.layers import initializers
import cPickle
import numpy as np


def load_h5(file):
	train_data = {}
	with h5py.File(file,'r') as hf:
		for k in hf.keys():
			tem = hf.get(k)
			train_data[k] = np.array(tem)
	return train_data

def load_json(file):
	fo = open(file, 'rb')
	dict = json.load(fo)
	fo.close()
	return dict

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_h5py(file, key=None):
	if key != None:
		with h5py.File(file,'r') as hf:
			data = hf.get(key)
			return np.asarray(data)
	else:
		print '[-] Can not load file'
