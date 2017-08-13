from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from highway import *
import pdb

class D_pretrained():
    def __init__(self, sess, dataset, negative_dataset, conf=None):

        self.sess = sess
        self.batch_size = conf.batch_size
        self.max_iter = conf.max_iter
        self.num_train = dataset.num_train
        self.hidden_size = conf.D_hidden_size     # 512
	self.max_to_keep = conf.max_to_keep
        self.dict_size = dataset.dict_size
        self.max_words = dataset.max_words
	self.lstm_steps = self.max_words+1
	self.img_dims = dataset.img_dims

        self.dataset = dataset
	self.negative_dataset = negative_dataset
	self.checkpoint_dir = conf.checkpoint_dir
	self.START = self.dataset.word2ix[u'<BOS>']
        self.END = self.dataset.word2ix[u'<EOS>']
        self.UNK = self.dataset.word2ix[u'<UNK>']
        self.NOT = self.dataset.word2ix[u'<NOT>']

	self.global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
        self.optim = tf.train.AdamOptimizer(conf.learning_rate)

	# placeholder
	self.fake_images = tf.placeholder(tf.float32, [self.batch_size, self.img_dims], name="fake_images")
	self.wrong_images = tf.placeholder(tf.float32, [self.batch_size, self.img_dims], name="wrong_images")
	self.right_images = tf.placeholder(tf.float32, [self.batch_size, self.img_dims], name="right_images")

	self.fake_text = tf.placeholder(tf.int32, [self.batch_size, self.max_words], name="fake_text")
	self.wrong_text = tf.placeholder(tf.int32, [self.batch_size, self.max_words], name="wrong_text")
	self.right_text = tf.placeholder(tf.int32, [self.batch_size, self.max_words], name="right_text")

	self.fake_length = tf.placeholder(tf.int32, [self.batch_size], name="fake_length")
	self.wrong_length = tf.placeholder(tf.int32, [self.batch_size], name="wrong_length")
	self.right_length = tf.placeholder(tf.int32, [self.batch_size], name="right_length")
	
	# build graph
	self.D_fake, D_fake_logits = self.build_Discriminator(self.fake_images, self.fake_text, self.fake_length, 
									name="D", reuse=False)
	self.D_wrong, D_wrong_logits = self.build_Discriminator(self.wrong_images, self.wrong_text, self.wrong_length, 
									name="D", reuse=True)
	self.D_right, D_right_logits = self.build_Discriminator(self.right_images, self.right_text, self.right_length, 
									name="D", reuse=True)
	# loss
	self.D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logits, tf.zeros_like(self.D_fake)))
	self.D_wrong_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_wrong_logits, tf.zeros_like(self.D_wrong)))
	self.D_right_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_right_logits, tf.ones_like(self.D_right)))
	self.loss = self.D_fake_loss+self.D_wrong_loss+self.D_right_loss
	# Summary
	self.D_fake_loss_sum = tf.scalar_summary("fake_loss", self.D_fake_loss)
	self.D_wrong_loss_sum = tf.scalar_summary("wrong_loss", self.D_wrong_loss)
	self.D_right_loss_sum = tf.scalar_summary("right_loss", self.D_right_loss)
	self.loss_sum = tf.scalar_summary("train_loss", self.loss)
	
	self.D_params_dict = {}
	params = tf.trainable_variables()
	for param in params:
            self.D_params_dict.update({param.name:param})

    def build_Discriminator(self, images, text, length, name="discriminator", reuse=False):

        ### sentence: B, S
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
	    with tf.variable_scope("lstm"):
		lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True)
            with tf.device('/cpu:0'), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.dict_size, self.hidden_size], "float32", random_uniform_init)
	    with tf.variable_scope("text_emb"):
                text_W = tf.get_variable("text_W", [2*self.hidden_size, self.hidden_size],"float32", random_uniform_init)
                text_b = tf.get_variable("text_b", [self.hidden_size], "float32", random_uniform_init)
	    with tf.variable_scope("images_emb"):
                images_W = tf.get_variable("images_W", [self.img_dims, self.hidden_size],"float32", random_uniform_init)
                images_b = tf.get_variable("images_b", [self.hidden_size], "float32", random_uniform_init)
	    with tf.variable_scope("scores_emb"):
                # "generator/scores"
                scores_W = tf.get_variable("scores_W", [self.hidden_size, 1], "float32", random_uniform_init)
                scores_b = tf.get_variable("scores_b", [1], "float32", random_uniform_init)

	    state = lstm1.zero_state(self.batch_size, 'float32')
	    start_token = tf.constant(self.START, dtype=tf.int32, shape=[self.batch_size])
	    # VQA use states
	    state_list = []
	    for j in range(self.lstm_steps):
		if j > 0:
		    tf.get_variable_scope().reuse_variables()
		with tf.device('/cpu:0'):
		    if j ==0:
			lstm1_in = tf.nn.embedding_lookup(word_emb_W, start_token)
		    else:
		    	lstm1_in = tf.nn.embedding_lookup(word_emb_W, text[:,j-1])
		with tf.variable_scope("lstm"):
                    # "generator/lstm"
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())     # output: B,H
		# apppend state from index 1 (the start of the word)
		if j > 0:
		    state_list.append(tf.concat(1,[state[0], state[1]]))

	    state_list = tf.pack(state_list)	# S,B,2H
	    state_list = tf.transpose(state_list, [1,0,2])	# B,S,2H
	    state_flatten = tf.reshape(state_list, [-1, 2*self.hidden_size])       # B*S, 2H
	    # length-1 => index start from 0
	    idx = tf.range(self.batch_size)*self.max_words + (length-1)  # B
	    state_gather = tf.gather(state_flatten, idx)	# B, 2H

	    # text embedding
	    text_emb = tf.matmul(state_gather, text_W) + text_b	# B,H
	    text_emb = tf.nn.tanh(text_emb)
	    # images embedding
	    images_emb = tf.matmul(images, images_W) + images_b	# B,H
	    images_emb = tf.nn.tanh(images_emb)
	    # embed to score
	    logits = tf.mul(text_emb, images_emb)	# B,H
	    score = tf.matmul(logits, scores_W) + scores_b

	    return tf.nn.sigmoid(score), score

    def train(self):

	self.train_op = self.optim.minimize(self.loss, global_step=self.global_step)
        self.writer = tf.train.SummaryWriter("./logs/D_pretrained", self.sess.graph)
	self.summary_op = tf.merge_all_summaries()
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(var_list=self.D_params_dict, max_to_keep=self.max_to_keep)
        count = 0
	for idx in range(self.max_iter//3000):
            self.save(self.checkpoint_dir, count)
            self.evaluate('test', count)
	    self.evaluate('train', count)
            for k in tqdm(range(3000)):
		right_images, right_text, _ = self.dataset.sequential_sample(self.batch_size)
		right_length = np.sum((right_text!=self.NOT)+0, 1)
		fake_images, fake_text, _ = self.negative_dataset.sequential_sample(self.batch_size)
		fake_length = np.sum((fake_text!=self.NOT)+0, 1)
		wrong_text = self.dataset.get_wrong_text(self.batch_size)
		wrong_length = np.sum((wrong_text!=self.NOT)+0, 1)
		feed_dict = {self.right_images:right_images, self.right_text:right_text, self.right_length:right_length, 
				self.fake_images:fake_images, self.fake_text:fake_text, self.fake_length:fake_length, 
				self.wrong_images:right_images, self.wrong_text:wrong_text, self.wrong_length:wrong_length}
		_, loss, summary_str = self.sess.run([self.train_op, self.loss, self.summary_op], feed_dict)
		self.writer.add_summary(summary_str, count)
                count += 1

    def evaluate(self, split, count):
	
	if split == 'test':
	    num_test_pair = -1
	elif split == 'train':
	    num_test_pair = 5000
	right_images, right_text, _ = self.dataset.get_paired_data(num_test_pair, phase=split)
	# the true paired data we get
	num_test_pair = len(right_images)
	fake_images, fake_text, _ = self.negative_dataset.get_paired_data(num_test_pair, phase=split)
	random_idx = range(num_test_pair)
	np.random.shuffle(random_idx)
	wrong_text = np.squeeze(right_text[random_idx, :])
	D_right_loss_t = []
	D_fake_loss_t = []
	D_wrong_loss_t = []
	D_right_acc_t = []
	D_fake_acc_t = []
	D_wrong_acc_t = []
	count = 0.
        for i in range(num_test_pair//self.batch_size):
	    right_images_batch = right_images[i*self.batch_size:(i+1)*self.batch_size,:]
	    fake_images_batch = fake_images[i*self.batch_size:(i+1)*self.batch_size,:]
	    right_text_batch = right_text[i*self.batch_size:(i+1)*self.batch_size,:]
	    fake_text_batch = fake_text[i*self.batch_size:(i+1)*self.batch_size,:]
	    wrong_text_batch = wrong_text[i*self.batch_size:(i+1)*self.batch_size,:]
	    right_length_batch = np.sum((right_text_batch!=self.NOT)+0, 1)
	    fake_length_batch = np.sum((fake_text_batch!=self.NOT)+0, 1)
	    wrong_length_batch = np.sum((wrong_text_batch!=self.NOT)+0, 1)
	    feed_dict = {self.right_images:right_images_batch, self.right_text:right_text_batch, 
			self.right_length:right_length_batch, self.fake_images:fake_images_batch, 
			self.fake_text:fake_text_batch, self.fake_length:fake_length_batch, 
			self.wrong_images:right_images_batch, self.wrong_text:wrong_text_batch, 
			self.wrong_length:wrong_length_batch}
	    D_right, D_fake, D_wrong, D_right_loss, D_fake_loss, D_wrong_loss = self.sess.run([self.D_right, self.D_fake, 
					self.D_wrong, self.D_right_loss, self.D_fake_loss, self.D_wrong_loss], feed_dict)
	    D_right_loss_t.append(D_right_loss)
	    D_fake_loss_t.append(D_fake_loss)
	    D_wrong_loss_t.append(D_wrong_loss)
	    D_right_acc_t.append(np.sum((D_right>0.5)+0))
	    D_fake_acc_t.append(np.sum((D_fake<0.5)+0))
	    D_wrong_acc_t.append(np.sum((D_wrong<0.5)+0))
	    count += self.batch_size

	print "Phase =", split.capitalize()
	print "======================= Loss ====================="
	print '[$] Right Pair Loss =', sum(D_right_loss_t)/count
	print '[$] Wrong Pair Loss =', sum(D_wrong_loss_t)/count
	print '[$] Fake Pair Loss =', sum(D_fake_loss_t)/count
	print "======================= Acc ======================"
	print '[$] Right Pair Acc. =', sum(D_right_acc_t)/count
	print '[$] Wrong Pair Acc. =', sum(D_wrong_acc_t)/count
	print '[$] Fake Pair Acc. =', sum(D_fake_acc_t)/count

    def save(self, checkpoint_dir, step):
        model_name = "D_Pretrained"
        model_dir = "%s" % (self.dataset.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir, "D_pretrained")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s" % (self.dataset.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

