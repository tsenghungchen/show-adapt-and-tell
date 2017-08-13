import json
import string
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from random import shuffle, seed

input_json = 'neuraltalk2/coco/coco_raw.json'
with open(input_json) as data_file:
    data = json.load(data_file)

seed(123)
shuffle(data)

skip_num = 0
val_data = {}
test_data = {}
train_data_ = {}

train_data = []

val_ann = []

val_dataset = []
test_dataset = []
train_dataset = []

counter = 0

for i in tqdm(range(len(data))):
    if i < 5000:
	# For GT
	idx = data[i]['id']
	caps = []
        for j in range(len(data[i]['captions'])):
            sen = data[i]['captions'][j].lower()
            for punc in string.punctuation:
                if punc in sen:
                    sen = sen.replace(punc, '')
            tmp = {}
            tmp['img_id'] = data[i]['id']
	    tmp['cap_id'] = j
            tmp['caption'] = sen
	    caps.append(tmp)

	val_data[idx] = caps

	# For load
        tmp = {}
        tmp['file_id'] = data[i]['file_path'].split('/')[1].split('.')[0]
        tmp['img_id'] = idx
        val_dataset.append(tmp)

    elif i < 10000:
        idx = data[i]['id']
        caps = []
        for j in range(len(data[i]['captions'])):
            sen = data[i]['captions'][j].lower()
            for punc in string.punctuation:
                if punc in sen:
                    sen = sen.replace(punc, '')
            tmp = {}
            tmp['img_id'] = data[i]['id']
            tmp['cap_id'] = j
            tmp['caption'] = sen
            caps.append(tmp)

        test_data[idx] = caps

        tmp = {}
        tmp['file_id'] = data[i]['file_path'].split('/')[1].split('.')[0]
        tmp['img_id'] = idx
        test_dataset.append(tmp)


    else:
        idx = data[i]['id']
        caps = []
        for j in range(len(data[i]['captions'])):
            sen = data[i]['captions'][j].lower()
            for punc in string.punctuation:
                if punc in sen:
                    sen = sen.replace(punc, '')



            tmp = {}
            tmp['img_id'] = data[i]['id']
            tmp['cap_id'] = j
            tmp['caption'] = sen
            caps.append(tmp)

        train_data_[idx] = caps

        tmp = {}
        tmp['file_id'] = data[i]['file_path'].split('/')[1].split('.')[0]
        tmp['img_id'] = idx
        train_dataset.append(tmp)



	# FOR TRAINING
        for j in range(len(data[i]['captions'])):   
	    sen = data[i]['captions'][j].lower()

	    for punc in string.punctuation:
	        if punc in sen:
		    sen = sen.replace(punc, '')    

	    if len(sen.split()) > 30:
	        skip_num += 1
	        continue

	    tmp = {}
	    tmp['file_id'] = data[i]['file_path'].split('/')[1].split('.')[0]
	    tmp['img_id'] = data[i]['id']
	    tmp['caption'] = sen
	    tmp['length'] = len(sen.split())
	    train_data.append(tmp)

print 'number of skip train data: ' + str(skip_num)

[u'info', u'images', u'licenses', u'type', u'annotations']

#json.dump(val_data, open('K_val_train.json', 'w'))
json.dump(val_data, open('./mscoco_data/K_val_annotation.json', 'w'))
json.dump(test_data, open('./mscoco_data/K_test_annotation.json', 'w'))
json.dump(train_data_, open('./mscoco_data/K_train_annotation.json', 'w'))

#json.dump(train_data, open('K_train_raw.json', 'w'))

json.dump(val_dataset, open('./mscoco_data/K_val_data.json', 'w'))
json.dump(test_dataset, open('./mscoco_data/K_test_data.json', 'w'))
json.dump(train_dataset, open('./mscoco_data/K_train_data.json', 'w'))
