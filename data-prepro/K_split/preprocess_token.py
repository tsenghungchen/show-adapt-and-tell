import re
import json
import numpy as np
from tqdm import tqdm
import pdb
import os
import pickle
import cPickle
import string
import sys

def unpickle(p):
	return cPickle.load(open(p,'r'))

def load_json(p):
	return json.load(open(p,'r'))

def clean_words(data):
	dict = {}
	freq = {}
	# start with 1
	idx = 1
	sentence_count = 0
	eliminate = 0
	max_w = 30
	for k in tqdm(range(len(data['caption_entity']))):
		sen = data['caption_entity'][k]
		filename = data['file_name'][k]
		# skip the no image description
		words = re.split(' ', sen)
		# pop the last u'.'
		n = len(words)
		if "" in words:
		    words.remove("")
		if n <= max_w:
			sentence_count += 1
			for word in words:
				if "\n" in word:
                                        word = word.replace("\n", "")
				for p in string.punctuation:
					if p in word:
						word = word.replace(p,'')
				word = word.lower()
				if word not in dict.keys():
					dict[word] = idx
					idx += 1
					freq[word] = 1
				else:
					freq[word] += 1
		else:
			eliminate += 1
	print 'Threshold(max_words) =', max_w
	print 'Eliminate =', eliminate 
	print 'Total sentence_count =', sentence_count
	print 'Number of different words =', len(dict.keys())
	print 'Saving....'
	np.savez('K_cleaned_words', dict=dict, freq=freq)
	return dict, freq

phase = sys.argv[1]
data_path = '../mscoco_data/K_annotation_'+phase+'2014.pkl'
data = unpickle(data_path)
thres = 5
if not os.path.isfile('../mscoco_data/dictionary_'+str(thres)+'.npz'):
	# clean the words through the frequency
	if not os.path.isfile('K_cleaned_words.npz'):
		dict, freq = clean_words(data)
	else:
		words = np.load('K_cleaned_words.npz')
		dict = words['dict'].item(0)
		freq = words['freq'].item(0)
	idx2word = {}
	word2idx = {}
	idx = 1
	for k in tqdm(dict.keys()):
		if freq[k] >= thres and k != "":
			word2idx[k] = idx
			idx2word[str(idx)] = k
			idx += 1

	word2idx[u'<BOS>'] = 0
	idx2word["0"] = u'<BOS>'
	word2idx[u'<EOS>'] = len(word2idx.keys())
	idx2word[str(len(idx2word.keys()))] = u'<EOS>'
	word2idx[u'<UNK>'] = len(word2idx.keys())
	idx2word[str(len(idx2word.keys()))] = u'<UNK>'
	word2idx[u'<NOT>'] = len(word2idx.keys())
        idx2word[str(len(idx2word.keys()))] = u'<NOT>'
	print 'Threshold of word fequency =', thres
	print 'Total words in the dictionary =', len(word2idx.keys())
	np.savez('../mscoco_data/dictionary_'+str(thres), word2idx=word2idx, idx2word=idx2word)
else:
	tem = np.load('../mscoco_data/dictionary_'+str(thres)+'.npz')
	word2idx = tem['word2idx'].item(0)
	idx2word = tem['idx2word'].item(0)

num_sentence = 0
eliminate = 0
tokenized_caption_list = []
caption_list = []
filename_list = []
caption_length = []
for k in tqdm(range(len(data['caption_entity']))):
	sen = data['caption_entity'][k]
	filename = data['file_name'][k]
	# skip the no image description
	words = re.split(' ', sen)
	# pop the last u'.'
	tokenized_sent = np.zeros([30+1], dtype=int)
	tokenized_sent.fill(int(word2idx[u'<NOT>']))
	#tokenized_sent[0] = int(word2idx[u'<BOS>'])
	valid = True
	count = 0
	caption = []
	
	if len(words) <= 30:
		for word in words:
			try:
				word = word.lower()
				for p in string.punctuation:
                                        if p in word:
                                                word = word.replace(p,'')
				if word != "":
					idx = int(word2idx[word])
					tokenized_sent[count] = idx
					caption.append(word)
					count += 1
			except KeyError:
				# if contain <UNK> then drop the sentence
				if phase == 'train':
					valid = False
					break
				else:
					tokenized_sent[count] = int(word2idx[u'<UNK>'])
					count += 1
		if valid:
			tokenized_sent[count] = (word2idx["<EOS>"])
			caption_list.append(caption)
			length = np.sum((tokenized_sent!=0)+0)
			tokenized_caption_list.append(tokenized_sent)
			filename_list.append(filename)
			caption_length.append(length)
			num_sentence += 1
		else:
			if phase == 'val':
				pdb.set_trace()
			eliminate += 1	
tokenized_caption_info = {}
tokenized_caption_info['caption_length'] = np.asarray(caption_length)
tokenized_caption_info['tokenized_caption_list'] = np.asarray(tokenized_caption_list)
tokenized_caption_info['caption_list'] = np.asarray(caption_list)
tokenized_caption_info['filename_list'] = np.asarray(filename_list)
print 'Number of sentence =', num_sentence
with open('../mscoco_data/tokenized_'+phase+'_caption.pkl', 'w') as outfile:
	pickle.dump(tokenized_caption_info, outfile)

