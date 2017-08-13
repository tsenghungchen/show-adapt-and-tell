import numpy as np
import os
import cPickle

# generate name2id & id2name dictionary
name_id_path = '../images.txt'
name_id = open(name_id_path).read().splitlines()
name2id = {}
id2name = {}
for img in name_id:
    name2id[img.split(' ')[1]] = img.split(' ')[0]
    id2name[img.split(' ')[0]] = img.split(' ')[1]

cPickle.dump(name2id, open('name2id.pkl', 'wb'))
cPickle.dump(id2name, open('id2name.pkl', 'wb'))

# generate id2caption dictionary for all images
# please download caption data on https://github.com/reedscot/cvpr2016. 
# CUB_CVPR16 will be created after unzipping. 
caption_path = '../CUB_CVPR16/text_c10/'
id2caption = {}
for name in name2id:
    txt_name = '.'.join(name.split('.')[0:-1]) + '.txt'
    txt_path = os.path.join(caption_path, txt_name)
    id = name2id[name]
    id2caption[id] = open(txt_path).read().splitlines()

cPickle.dump(id2caption, open('id2caption.pkl', 'wb'))

# generate split dictionary
train_path = './ECCV16_explanations_splits/train_noCub.txt'
test_path = './ECCV16_explanations_splits/test.txt'
val_path = './ECCV16_explanations_splits/val.txt'
splits = {}
splits['train_name'] = open(train_path).read().splitlines()
splits['test_name'] = open(test_path).read().splitlines()
splits['val_name'] = open(val_path).read().splitlines()

splits['train_id'] = [name2id[n] for n in splits['train_name']]
splits['test_id'] = [name2id[n] for n in splits['test_name']]
splits['val_id'] = [name2id[n] for n in splits['val_name']]

cPickle.dump(splits, open('splits.pkl', 'wb'))

