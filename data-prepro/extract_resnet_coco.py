import sys
sys.path.append('/home/PaulChen/deep-residual-networks/caffe/python')
import caffe
import numpy as np
import argparse
import cv2
import os, time
import json
import pdb
import PIL
from tqdm import tqdm
from PIL import Image
import re
import pickle as pk

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Extract a CNN features')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def set_transformer(net):
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data',(2,0,1))
	transformer.set_mean('data', np.load(\
		os.path.join('resnet_model','ResNet_mean.npy')))
	transformer.set_input_scale('data', 255)
	return transformer

def iter_frames(im):
 	try:
 		i= 0
 		while 1:
 			im.seek(i)
 			imframe = im.copy()
 			if i == 0: 
 				palette = imframe.getpalette()
 			else:
 				imframe.putpalette(palette)
 			yield imframe
 			i += 1
 	except EOFError:
 		pass

def extract_image(net, image_file):
	batch_size = 1
	transformer = set_transformer(net)
	if image_file.split('.')[-1] == 'gif':
		img = Image.open(image_file).convert("P",palette=Image.ADAPTIVE, colors=256)
		newfile = ''.join(image_file.split('.')[:-1])+'.png'
		for i, frame in enumerate(iter_frames(img)):
			frame.save(newfile,**frame.info)
		image_file = newfile
	
	img = cv2.imread(image_file)
	img = img.astype('float') / 255
	net.blobs['data'].data[:] = transformer.preprocess('data', img)
	net.forward()
	blobs_out_pool5 = net.blobs['pool5'].data[0,:,0,0]
	return blobs_out_pool5


def split(split, net, feat_dict):
        print 'load ' + split 
	img_dir = './coco/'
	img_path = os.path.join(img_dir, split)
	img_list = os.listdir(img_path)
	pool5_list = []
	prob_list = []
        for k in tqdm(img_list):
		blobs_out_pool5 = extract_image(net, os.path.join(img_path,k))
		feat_dict[k.split('.')[0]] = np.array(blobs_out_pool5)

	return feat_dict

if __name__ == '__main__':
	args = parse_args()
	caffe_path = os.path.join('/home','PaulChen','caffe','python')

	print 'caffe setting'
	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)

	print 'load caffe'
	net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
	net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

	feat_dict = {}
	split('train2014', net, feat_dict)
	split('val2014', net, feat_dict)
	pk.dump(feat_dict, open('./mscoco_data/coco_trainval_feat.pkl','w'))

