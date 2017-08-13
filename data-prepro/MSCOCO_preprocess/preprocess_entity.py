import re
import pickle
import json
import numpy as np
from tqdm import tqdm
import pdb
import sys
def load_json(p):
	return json.load(open(p,'r'))

desired_phase = sys.argv[1]
split_path = 'K_split.json'
split = load_json(split_path)
split_id = split[desired_phase]

phase = ['train', 'val']
id2name = {}
name2id = {}
id2caption = {}
description_list = []
img_name = []
for p in phase:
	data_path = './annotations/captions_%s2014.json' % p
	data = load_json(data_path)
	for img_info in data['images']:
		if img_info['id'] in split_id:
			id2name[str(img_info['id'])] = img_info['file_name']
			name2id[img_info['file_name']] = str(img_info['id'])
			id2caption[str(img_info['id'])] = []
	count =	0
	for k in tqdm(range(len(data['annotations']))):
		sen = data['annotations'][k]['caption']
		image_id = data['annotations'][k]['image_id']
		if image_id in split_id:
			id2caption[str(image_id)].append(sen)
			file_name = id2name[str(image_id)]
			description_list.append(sen)
			img_name.append(file_name)

out = {}
out['caption_entity'] = description_list
out['file_name'] = img_name
out['id2filename'] = id2name
out['filename2id'] = name2id
out['id2caption'] = id2caption
print 'Saving ...'
print 'Numer of sentence =', len(description_list)	
with open('./mscoco_data/K_annotation_%s2014.pkl'%desired_phase, 'w') as outfile:
	pickle.dump(out, outfile)

