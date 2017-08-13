import json
import string
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from random import shuffle, seed
import pickle as pk
import pdb
input_data = 'split.pkl'
with open(input_data) as data_file:
    dataset = pk.load(data_file)

skip_num = 0
val_data = {}
test_data = {}
train_data = []

val_dataset = []
test_dataset = []
counter = 0
id2name = pk.load(open('id2name.pkl'))
data = pk.load(open('id2caption.pkl'))

for i in dataset['test_id']:
        caps = []
        # For GT
        name = id2name[i]
        count = 0
        for sen in data[i]:
            for punc in string.punctuation:
                if punc in sen:
                    sen = sen.replace(punc, '')
                    
            tmp = {}
            tmp['filename'] = name
            tmp['img_id'] = i
            tmp['cap_id'] = count
            tmp['caption'] = sen
            count += 1
            caps.append(tmp)

        test_data[i] = caps
print 'number of skip train data: ' + str(skip_num)
[u'info', u'images', u'licenses', u'type', u'annotations']
json.dump(test_data, open('cub_data/K_test_annotation.json', 'w'))
