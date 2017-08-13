from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from highway import *
import copy 
from coco_caption.pycocoevalcap.eval import COCOEvalCap
import pdb

def calculate_loss_and_acc_with_logits(predictions, logits, label, l2_loss, l2_reg_lambda):
    # Calculate Mean cross-entropy loss
    with tf.variable_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(tf.squeeze(logits), label)
        D_loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
    with tf.variable_scope("accuracy"):
        correct_predictions = tf.equal(predictions, tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
    return D_loss, accuracy


class SeqGAN():
    def __init__(self, sess, dataset, D_info, conf=None):
	self.sess = sess
	self.model_name = conf.model_name
        self.batch_size = conf.batch_size
        self.max_iter = conf.max_iter
	self.max_to_keep = conf.max_to_keep
	self.is_train = conf.is_train
	# Testing => dropout rate is 0
	if self.is_train:
	    self.drop_out_rate = conf.drop_out_rate
	else:
	    self.drop_out_rate = 0

	self.num_train = dataset.num_train
        self.G_hidden_size = conf.G_hidden_size     	# 512
	self.D_hidden_size = conf.D_hidden_size		# 512
        self.dict_size = dataset.dict_size
        self.max_words = dataset.max_words
        self.dataset = dataset
	self.img_dims = self.dataset.img_dims
	self.checkpoint_dir = conf.checkpoint_dir
	self.lstm_steps = self.max_words+1
        self.START = self.dataset.word2ix[u'<BOS>']
        self.END = self.dataset.word2ix[u'<EOS>']
        self.UNK = self.dataset.word2ix[u'<UNK>']
        self.NOT = self.dataset.word2ix[u'<NOT>']
	self.method = conf.method
	self.discount = conf.discount
	self.load_pretrain = conf.load_pretrain
	self.filter_sizes = D_info['filter_sizes']
        self.num_filters = D_info['num_filters']
        self.num_filters_total = sum(self.num_filters)
        self.num_classes = D_info['num_classes']
	self.num_domains = 3
        self.l2_reg_lambda = D_info['l2_reg_lambda']

	
	# D placeholder
	self.images = tf.placeholder('float32', [self.batch_size, self.img_dims])
	self.right_text = tf.placeholder('int32', [self.batch_size, self.max_words])
	self.wrong_text = tf.placeholder('int32', [self.batch_size, self.max_words])
	self.wrong_length = tf.placeholder('int32', [self.batch_size], name="wrong_length")
        self.right_length = tf.placeholder('int32', [self.batch_size], name="right_length")

	# Domain Classider
	self.src_images = tf.placeholder('float32', [self.batch_size, self.img_dims])
	self.tgt_images = tf.placeholder('float32', [self.batch_size, self.img_dims])
	self.src_text = tf.placeholder('int32', [self.batch_size, self.max_words])
	self.tgt_text = tf.placeholder('int32', [self.batch_size, self.max_words])
	# Optimizer
	self.G_optim = tf.train.AdamOptimizer(conf.learning_rate)
	self.D_optim = tf.train.AdamOptimizer(conf.learning_rate)
	self.T_optim = tf.train.AdamOptimizer(conf.learning_rate)
	self.Domain_image_optim = tf.train.AdamOptimizer(conf.learning_rate)
	self.Domain_text_optim = tf.train.AdamOptimizer(conf.learning_rate)
        D_info["sentence_length"] = self.max_words
        self.D_info = D_info

	###################################################
        # Generator                                       #
        ###################################################
	# G placeholder
	state_list, predict_words_list_sample, log_probs_action_picked_list, self.rollout_mask, self.predict_mask = self.generator(name='G', reuse=False)
	predict_words_sample = tf.pack(predict_words_list_sample)
        self.predict_words_sample = tf.transpose(predict_words_sample, [1,0]) # B,S
	# for testing
	# argmax prediction
        _, predict_words_list_argmax, log_probs_action_picked_list_argmax, _, self.predict_mask_argmax = self.generator_test(name='G', reuse=True)
        predict_words_argmax = tf.pack(predict_words_list_argmax)
        self.predict_words_argmax = tf.transpose(predict_words_argmax, [1,0]) # B,S
	rollout = []
	rollout_length = []
	rollout_num = 3
	for i in range(rollout_num):
	    rollout_i, rollout_length_i = self.rollout(predict_words_list_sample, state_list, name="G")      # S*B, S
	    rollout.append(rollout_i)    # R,B,S
	    rollout_length.append(rollout_length_i) # R,B, 1
	   
	rollout = tf.pack(rollout)      # R,B,S
	rollout = tf.reshape(rollout, [-1, self.max_words])     # R*B,S
	rollout_length = tf.pack(rollout_length)    # R,B,1
	rollout_length = tf.reshape(rollout_length, [-1, 1])     # R*B, 1
	rollout_length = tf.squeeze(rollout_length)
	rollout_size = self.batch_size * self.max_words * rollout_num
	images_expand = tf.expand_dims(self.images, 1)  # B,1,I
	images_tile = tf.tile(images_expand, [1, self.max_words, 1])    # B,S,I
	images_tile_transpose = tf.transpose(images_tile, [1,0,2])      # S,B,I
	images_tile_transpose = tf.tile(tf.expand_dims(images_tile_transpose, 0), [rollout_num,1,1,1])  #R,S,B,I
	images_reshape = tf.reshape(images_tile_transpose, [-1, self.img_dims]) #R*S*B,I

	D_rollout_vqa_softmax, D_rollout_logits_vqa = self.discriminator(rollout_size, images_reshape, rollout, rollout_length, name="D", reuse=False)
	D_rollout_text, D_rollout_text_softmax, D_logits_rollout_text, l2_loss_rollout_text = self.text_discriminator(rollout, D_info, name="D_text", reuse=False)
	reward = tf.multiply(D_rollout_vqa_softmax[:,0], D_rollout_text_softmax[:,0]) # S*B, 1

	reward = tf.reshape(reward, [rollout_num, -1])  # R, S*B
	reward = tf.reduce_mean(reward, 0)      # S*B

        self.rollout_reward = tf.reshape(reward, [self.max_words, self.batch_size])      # S,B
        D_logits_rollout_reshape = tf.reshape(self.rollout_reward, [-1])
        self.G_loss = (-1)*tf.reduce_sum(log_probs_action_picked_list*tf.stop_gradient(D_logits_rollout_reshape)) / tf.reduce_sum(tf.stop_gradient(self.predict_mask))

	# Teacher Forcing
        self.mask = tf.placeholder('float32', [self.batch_size, self.max_words])       # mask out the loss
	self.teacher_loss, self.teacher_loss_sum = self.Teacher_Forcing(self.right_text, self.mask, name="G", reuse=True)

        ###################################################
        # Discriminator                                   #
        ###################################################
	# take the sample as fake data
        D_info["sentence_length"] = self.max_words

	# take the argmax sample as fake data
	self.fake_length = tf.reduce_sum(tf.stop_gradient(self.predict_mask),1)
	D_fake_vqa_softmax, D_fake_logits_vqa = self.discriminator(self.batch_size, self.images, tf.to_int32(self.predict_words_sample), tf.to_int32(self.fake_length), name="D", reuse=True)
	D_right_vqa_softmax, D_right_logits_vqa = self.discriminator(self.batch_size, self.images, self.right_text, 
                                                        self.right_length, name="D", reuse=True)
	D_wrong_vqa_softmax, D_wrong_logits_vqa = self.discriminator(self.batch_size, self.images, self.wrong_text,
                                                        self.wrong_length, name="D", reuse=True)

	D_right_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(D_right_logits_vqa, 
		tf.concat(1,(tf.ones((self.batch_size,1)), tf.zeros((self.batch_size,1)), tf.zeros((self.batch_size,1))))))
	D_wrong_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(D_wrong_logits_vqa,
		tf.concat(1,(tf.zeros((self.batch_size,1)), tf.ones((self.batch_size,1)), tf.zeros((self.batch_size,1))))))
	D_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(D_fake_logits_vqa,
		tf.concat(1,(tf.zeros((self.batch_size,1)), tf.zeros((self.batch_size,1)), tf.ones((self.batch_size,1))))))


	self.D_loss = D_fake_loss + D_right_loss + D_wrong_loss
	###################################################
	# Text Domain Classifier
	###################################################
	D_src_text, D_src_text_softmax, D_logits_src_text, l2_loss_src_text = self.text_discriminator(self.src_text, D_info, name="D_text", reuse=True)
	D_tgt_text, D_tgt_text_softmax, D_logits_tgt_text, l2_loss_tgt_text = self.text_discriminator(self.tgt_text, D_info, name="D_text", reuse=True)
	D_fake_text, D_fake_text_softmax, D_logits_fake_text, l2_loss_fake_text = self.text_discriminator(self.predict_words_sample, D_info, name="D_text", reuse=True)


	D_src_loss_text, D_src_acc_text = calculate_loss_and_acc_with_logits(D_src_text,
                                D_logits_src_text, tf.concat(1,(tf.zeros((self.batch_size,1)), tf.zeros((self.batch_size,1)),  
					tf.ones((self.batch_size,1)))), l2_loss_src_text, D_info["l2_reg_lambda"])
	D_fake_loss_text, D_fake_acc_text = calculate_loss_and_acc_with_logits(D_fake_text,
				D_logits_fake_text, tf.concat(1,(tf.zeros((self.batch_size,1)), tf.ones((self.batch_size,1)),
					tf.zeros((self.batch_size,1)))), l2_loss_fake_text, D_info["l2_reg_lambda"])
        D_tgt_loss_text, D_tgt_acc_text = calculate_loss_and_acc_with_logits(D_tgt_text,
                                D_logits_tgt_text, tf.concat(1,(tf.ones((self.batch_size,1)), tf.zeros((self.batch_size,1)),
                                        tf.zeros((self.batch_size,1)))), l2_loss_tgt_text, D_info["l2_reg_lambda"])
	self.D_text_loss = D_src_loss_text + D_tgt_loss_text + D_fake_loss_text


	########################## tensorboard summary:########################
        # D_real_sum, D_fake_sum = the sigmoid output
        # D_real_loss_sum, D_fake_loss_sum = the loss for different kinds input
        # D_loss_sum, G_loss_sum = loss of the G&D
        #######################################################################
	self.start_reward_sum = tf.scalar_summary("start_reward", tf.reduce_mean(self.rollout_reward[0,:]))
	self.total_reward_sum = tf.scalar_summary("total_mean_reward", tf.reduce_mean(self.rollout_reward))
	self.logprobs_mean_sum = tf.scalar_summary("logprobs_mean", tf.reduce_sum(log_probs_action_picked_list)/tf.reduce_sum(self.predict_mask))
	self.logprobs_dist_sum = tf.histogram_summary("log_probs", log_probs_action_picked_list)
	self.D_fake_loss_sum = tf.scalar_summary("D_fake_loss", D_fake_loss)
	self.D_wrong_loss_sum = tf.scalar_summary("D_wrong_loss", D_wrong_loss)
	self.D_right_loss_sum = tf.scalar_summary("D_right_loss", D_right_loss)
	self.D_loss_sum = tf.scalar_summary("D_loss", self.D_loss)
	self.G_loss_sum = tf.scalar_summary("G_loss", self.G_loss)
	###################################################
        # Record the paramters                            #
        ###################################################
	params = tf.trainable_variables()
	self.R_params = []
	self.G_params = []
	self.D_params = []
	self.G_params_dict = {}
	self.D_params_dict = {}
	for param in params:
	    if "R" in param.name:
		self.R_params.append(param)
	    elif "G" in param.name:
		self.G_params.append(param)
		self.G_params_dict.update({param.name:param})
	    elif "D" in param.name:
	    	self.D_params.append(param)
	    	self.D_params_dict.update({param.name:param})
	print "Build graph complete"

    def rollout_update(self):
	for r, g in zip(self.R_params, self.G_params):
	    assign_op = r.assign(g)
	    self.sess.run(assign_op)
    def discriminator(self, batch_size, images, text, length, name="discriminator", reuse=False):

        ### sentence: B, S
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.D_hidden_size, state_is_tuple=True)
		lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, output_keep_prob=1-self.drop_out_rate)
            with tf.device('/cpu:0'), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.dict_size, self.D_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("text_emb"):
                text_W = tf.get_variable("text_W", [2*self.D_hidden_size, self.D_hidden_size],"float32", random_uniform_init)
                text_b = tf.get_variable("text_b", [self.D_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("images_emb"):
                images_W = tf.get_variable("images_W", [self.img_dims, self.D_hidden_size],"float32", random_uniform_init)
                images_b = tf.get_variable("images_b", [self.D_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("scores_emb"):
                # "generator/scores"
                scores_W = tf.get_variable("scores_W", [self.D_hidden_size, 3], "float32", random_uniform_init)
                scores_b = tf.get_variable("scores_b", [3], "float32", random_uniform_init)

            state = lstm1.zero_state(batch_size, 'float32')
            start_token = tf.constant(self.START, dtype=tf.int32, shape=[batch_size])
            # VQA use states
            state_list = []
            for j in range(self.max_words+1):
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

            state_list = tf.pack(state_list)    # S,B,2H
            state_list = tf.transpose(state_list, [1,0,2])      # B,S,2H
            state_flatten = tf.reshape(state_list, [-1, 2*self.D_hidden_size])       # B*S, 2H
            # length-1 => index start from 0
	    # need to prevent length = 0
	    length_index = length-1
	    condition = tf.greater_equal(length_index, 0)	# B
	    length_index = tf.select(condition, length_index, tf.constant(0, dtype=tf.int32, shape=[batch_size]))
            idx = tf.range(batch_size)*self.max_words + length_index  # B
            state_gather = tf.gather(state_flatten, idx)        # B, 2H
            # text embedding
            text_emb = tf.matmul(state_gather, text_W) + text_b # B,H
            text_emb = tf.nn.tanh(text_emb)
            # images embedding
            images_emb = tf.matmul(images, images_W) + images_b # B,H
            images_emb = tf.nn.tanh(images_emb)
            # embed to score
            logits = tf.mul(text_emb, images_emb)       # B,H
            score = tf.matmul(logits, scores_W) + scores_b

            #return tf.nn.sigmoid(score), score
	    return tf.nn.softmax(score), score


    def text_discriminator(self, sentence, info, name="text_discriminator", reuse=False):
        ### sentence: B, S
        hidden_size = self.D_hidden_size
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.device('/cpu:0'), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.dict_size, hidden_size], "float32", random_uniform_init)
                embedded_chars = tf.nn.embedding_lookup(word_emb_W, sentence) # B,S,H
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)      # B,S,H,1
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [info["num_filters_total"], self.num_domains],
                                                "float32", random_uniform_init)
                output_b = tf.get_variable("output_b", [self.num_domains], "float32", random_uniform_init)
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)
            for filter_size, num_filter in zip(info["filter_sizes"], info["num_filters"]):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, hidden_size, 1, num_filter]
                    W = tf.get_variable("W", filter_shape, "float32", random_uniform_init)
                    b = tf.get_variable("b", [num_filter], "float32", random_uniform_init)
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, info["sentence_length"] - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            h_pool = tf.concat(3, pooled_outputs)      # B,1,1,total filters
            h_pool_flat = tf.reshape(h_pool, [-1, info["num_filters_total"]])        # b, total filters

            # Add highway
            with tf.variable_scope("highway"):
                h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)
            with tf.variable_scope("output"):
                l2_loss += tf.nn.l2_loss(output_W)
                l2_loss += tf.nn.l2_loss(output_b)
                logits = tf.nn.xw_plus_b(h_highway, output_W, output_b, name="logits")
                logits_softmax = tf.nn.softmax(logits)
                predictions = tf.argmax(logits_softmax, 1, name="predictions")
            return predictions, logits_softmax, logits, l2_loss

    def domain_classifier(self, images, name="G", reuse=False):	
	random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)	
	with tf.variable_scope(name):
	    tf.get_variable_scope().reuse_variables()
	    with tf.variable_scope("images"):
                # "generator/images"
                images_W = tf.get_variable("images_W", [self.img_dims, self.G_hidden_size], "float32", random_uniform_init)
		images_emb = tf.matmul(images, images_W)   	# B,H

        l2_loss = tf.constant(0.0)
	with tf.variable_scope("domain"):
	    if reuse:
		tf.get_variable_scope().reuse_variables()
	    with tf.variable_scope("output"):
	        output_W = tf.get_variable("output_W", [self.G_hidden_size, self.num_domains],
                                                "float32", random_uniform_init)
                output_b = tf.get_variable("output_b", [self.num_domains], "float32", random_uniform_init)
		l2_loss += tf.nn.l2_loss(output_W)
		l2_loss += tf.nn.l2_loss(output_b)
		logits = tf.nn.xw_plus_b(images_emb, output_W, output_b, name="logits")
		predictions = tf.argmax(logits, 1, name="predictions")

	    return predictions, logits, l2_loss
		

    def rollout(self, predict_words, state_list, name="R"):

	random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)	
	with tf.variable_scope(name):
	    tf.get_variable_scope().reuse_variables()
	    with tf.variable_scope("images"):
                # "generator/images"
                images_W = tf.get_variable("images_W", [self.img_dims, self.G_hidden_size], "float32", random_uniform_init)
	    with tf.variable_scope("lstm"):
                # WONT BE CREATED HERE
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
		lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, output_keep_prob=1-self.drop_out_rate)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                # "R/embedding"
                word_emb_W = tf.get_variable("word_emb_W",[self.dict_size, self.G_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("output"):
                # "R/output"
                output_W = tf.get_variable("output_W", [self.G_hidden_size, self.dict_size], "float32", random_uniform_init)
	    rollout_list = []
	    length_mask_list = []
	    # rollout for the first time step
	    for step in range(self.max_words):
		sample_words = predict_words[step]
		state = state_list[step]
		rollout_step_list = []
		mask = tf.constant(True, "bool", [self.batch_size])		
		# used to calcualte the length of the rollout sentence
		length_mask_step = []
		for j in range(step+1):
		    mask_out_word = tf.select(mask, predict_words[j], 
						tf.constant(self.NOT, dtype=tf.int64, shape=[self.batch_size]))
		    rollout_step_list.append(mask_out_word)
		    length_mask_step.append(mask)
		    prev_mask = mask
                    mask_step = tf.not_equal(predict_words[j], self.END)    # B
                    mask = tf.logical_and(prev_mask, mask_step)
		for j in range(self.max_words-step-1):
		    if step != 0 or j != 0:
	                tf.get_variable_scope().reuse_variables()
            	    with tf.device("/cpu:0"):
                	sample_words_emb = tf.nn.embedding_lookup(word_emb_W, tf.stop_gradient(sample_words))
  	            with tf.variable_scope("lstm"):
                	output, state = lstm1(sample_words_emb, state, scope=tf.get_variable_scope())     # output: B,H
		    logits = tf.matmul(output, output_W)
		    # add 1e-8 to prevent log(0) 
            	    log_probs = tf.log(tf.nn.softmax(logits)+1e-8)   # B,D
            	    sample_words = tf.squeeze(tf.multinomial(log_probs,1))
		    mask_out_word = tf.select(mask, sample_words, 
						tf.constant(self.NOT, dtype=tf.int64, shape=[self.batch_size]))
		    rollout_step_list.append(mask_out_word)
		    length_mask_step.append(mask)
                    prev_mask = mask
                    mask_step = tf.not_equal(sample_words, self.END)    # B
                    mask = tf.logical_and(prev_mask, mask_step)

		length_mask_step = tf.pack(length_mask_step)	# S,B
		length_mask_step = tf.transpose(length_mask_step, [1,0])	# B,S
		length_mask_list.append(length_mask_step)
		rollout_step_list = tf.pack(rollout_step_list)	# S,B
		rollout_step_list = tf.transpose(rollout_step_list, [1,0])	# B,S
		rollout_list.append(rollout_step_list)
	
	    length_mask_list = tf.pack(length_mask_list)	# S,B,S
	    length_mask_list = tf.reshape(length_mask_list, [-1, self.max_words])	# S*B,S
	    rollout_list = tf.pack(rollout_list)	# S,B,S
	    rollout_list = tf.reshape(rollout_list, [-1, self.max_words])	# S*B, S
	    rollout_length = tf.to_int32(tf.reduce_sum(tf.to_float(length_mask_list),1))
 	    return rollout_list, rollout_length

    def Teacher_Forcing(self, target_sentence, mask, name='generator', reuse=False):
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):
	    if reuse:
		tf.get_variable_scope().reuse_variables()
	    with tf.variable_scope("images"):
                # "generator/images"
                images_W = tf.get_variable("images_W", [self.img_dims, self.G_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("lstm"):
		# "generator/lstm"
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
		lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, output_keep_prob=1-self.drop_out_rate)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                # "generator/embedding"
                word_emb_W = tf.get_variable("word_emb_W", [self.dict_size, self.G_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("output"):
                # "generator/output"
                output_W = tf.get_variable("output_W", [self.G_hidden_size, self.dict_size], "float32", random_uniform_init)

	    start_token = tf.constant(self.START, dtype=tf.int32, shape=[self.batch_size])
	    state = lstm1.zero_state(self.batch_size, 'float32')
	    teacher_loss = 0.
	    for j in range(self.lstm_steps):
		if j == 0:
		    images_emb = tf.matmul(self.images, images_W)   	# B,H
		    lstm1_in = images_emb
		else:
		    tf.get_variable_scope().reuse_variables()
		    with tf.device("/cpu:0"):
			if j == 1:
			    # <BOS>
			    lstm1_in = tf.nn.embedding_lookup(word_emb_W, start_token)
			else:
			    # schedule sampling
			    lstm1_in = tf.nn.embedding_lookup(word_emb_W, target_sentence[:,j-2])

		with tf.variable_scope("lstm"):
		    # "generator/lstm"
		    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())     # output: B,H

		if j > 0:
		    logits = tf.matmul(output, output_W)       		# B,D
		    # calculate loss
		    log_probs = tf.log(tf.nn.softmax(logits)+1e-8)   # B,D
		    action_picked = tf.range(self.batch_size)*(self.dict_size) + target_sentence[:,j-1]
		    log_probs_action_picked = tf.mul(tf.gather(tf.reshape(log_probs, [-1]), action_picked), mask[:,j-1])
		    loss_t = (-1)*tf.reduce_sum(log_probs_action_picked*tf.ones(self.batch_size))
		    teacher_loss += loss_t

	    teacher_loss /= tf.reduce_sum(mask)
	    teacher_loss_sum = tf.scalar_summary("teacher_loss", teacher_loss)

	    return teacher_loss, teacher_loss_sum

    def generator(self, name='generator', reuse=False):

        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):
	    if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("images"):
                # "generator/images"
                images_W = tf.get_variable("images_W", [self.img_dims, self.G_hidden_size], "float32", random_uniform_init)
                #images_b = tf.get_variable("images_b", [self.G_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
                lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, output_keep_prob=1-self.drop_out_rate)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                # "generator/embedding"
                word_emb_W = tf.get_variable("word_emb_W", [self.dict_size, self.G_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("output"):
                # "generator/output"
                # dict size minus 1 => remove <UNK>
                output_W = tf.get_variable("output_W", [self.G_hidden_size, self.dict_size], "float32", random_uniform_init)

            start_token = tf.constant(self.START, dtype=tf.int32, shape=[self.batch_size])
            state = lstm1.zero_state(self.batch_size, 'float32')
	    mask = tf.constant(True, "bool", [self.batch_size])
	    log_probs_action_picked_list = []
	    predict_words = []
	    state_list = []
	    predict_mask_list = []
            for j in range(self.max_words+1):
                if j == 0:
                    #images_emb = tf.matmul(self.images, images_W) + images_b       # B,H
		    images_emb = tf.matmul(self.images, images_W)
                    lstm1_in = images_emb
                else:
                    tf.get_variable_scope().reuse_variables()
                    with tf.device("/cpu:0"):
                        if j == 1:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, tf.stop_gradient(sample_words))
                with tf.variable_scope("lstm"):
                    # "generator/lstm"
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())     # output: B,H
                if j > 0:
		    logits = tf.matmul(output, output_W)
		    log_probs = tf.log(tf.nn.softmax(logits)+1e-8)	# B,D
		    # word drawn from the multinomial distribution
		    sample_words = tf.reshape(tf.multinomial(log_probs,1), [self.batch_size])
		    mask_out_word = tf.select(mask, sample_words,
						tf.constant(self.NOT, dtype=tf.int64, shape=[self.batch_size]))
                    predict_words.append(mask_out_word)
		    #predict_words.append(sample_words)
		    # the mask should be dynamic
		    # if the sentence is: This is a dog <END>
		    # the predict_mask_list is: 1,1,1,1,1,0,0,.....
		    predict_mask_list.append(tf.to_float(mask))
		    action_picked = tf.range(self.batch_size)*(self.dict_size) + tf.to_int32(sample_words)        # B
		    # mask out the word beyond the <END>
		    log_probs_action_picked = tf.mul(tf.gather(tf.reshape(log_probs, [-1]), action_picked), tf.to_float(mask))
                    log_probs_action_picked_list.append(log_probs_action_picked)
                    prev_mask = mask
                    mask_step = tf.not_equal(sample_words, self.END)    # B
                    mask = tf.logical_and(prev_mask, mask_step)
                    state_list.append(state)

	    predict_mask_list = tf.pack(predict_mask_list)      # S,B
            predict_mask_list = tf.transpose(predict_mask_list, [1,0])  # B,S
            log_probs_action_picked_list = tf.pack(log_probs_action_picked_list)        # S,B
            log_probs_action_picked_list = tf.reshape(log_probs_action_picked_list, [-1])       # S*B
            return state_list, predict_words, log_probs_action_picked_list, None, predict_mask_list

    def generator_test(self, name='generator', reuse=False):

        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("images"):
                # "generator/images"
                images_W = tf.get_variable("images_W", [self.img_dims, self.G_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                # "generator/embedding"
                word_emb_W = tf.get_variable("word_emb_W", [self.dict_size, self.G_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("output"):
                # "generator/output"
                # dict size minus 1 => remove <UNK>
                output_W = tf.get_variable("output_W", [self.G_hidden_size, self.dict_size], "float32", random_uniform_init)
            start_token = tf.constant(self.START, dtype=tf.int32, shape=[self.batch_size])
            state = lstm1.zero_state(self.batch_size, 'float32')
            mask = tf.constant(True, "bool", [self.batch_size])
            log_probs_action_picked_list = []
            predict_words = []
            state_list = []
            predict_mask_list = []
            for j in range(self.max_words+1):
                if j == 0:
		    images_emb = tf.matmul(self.images, images_W)
                    lstm1_in = images_emb
                else:
                    tf.get_variable_scope().reuse_variables()
                    with tf.device("/cpu:0"):
                        if j == 1:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, tf.stop_gradient(sample_words))
                with tf.variable_scope("lstm"):
                    # "generator/lstm"
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())     # output: B,H
                if j > 0:
                    #logits = tf.matmul(output, output_W) + output_b       # B,D
		    logits = tf.matmul(output, output_W)
                    log_probs = tf.log(tf.nn.softmax(logits)+1e-8)      # B,D
                    # word drawn from the multinomial distribution
		    sample_words = tf.argmax(log_probs, 1)	# B
		    mask_out_word = tf.select(mask, sample_words,
                                                tf.constant(self.NOT, dtype=tf.int64, shape=[self.batch_size]))
                    predict_words.append(mask_out_word)
                    # the mask should be dynamic
                    # if the sentence is: This is a dog <END>
                    # the predict_mask_list is: 1,1,1,1,1,0,0,.....
                    predict_mask_list.append(tf.to_float(mask))
                    action_picked = tf.range(self.batch_size)*(self.dict_size) + tf.to_int32(sample_words)        # B
                    # mask out the word beyond the <END>
                    log_probs_action_picked = tf.mul(tf.gather(tf.reshape(log_probs, [-1]), action_picked), tf.to_float(mask))
                    log_probs_action_picked_list.append(log_probs_action_picked)
                    prev_mask = mask
                    mask_step = tf.not_equal(sample_words, self.END)    # B
                    mask = tf.logical_and(prev_mask, mask_step)
                    state_list.append(state)

            predict_mask_list = tf.pack(predict_mask_list)      # S,B
            predict_mask_list = tf.transpose(predict_mask_list, [1,0])  # B,S
            log_probs_action_picked_list = tf.pack(log_probs_action_picked_list)        # S,B
            log_probs_action_picked_list = tf.reshape(log_probs_action_picked_list, [-1])       # S*B
            return state_list, predict_words, log_probs_action_picked_list, None, predict_mask_list


    def train(self):

	self.G_train_op = self.G_optim.minimize(self.G_loss, var_list=self.G_params)
	self.G_hat_train_op = self.T_optim.minimize(self.teacher_loss, var_list=self.G_params)
	self.D_train_op = self.D_optim.minimize(self.D_loss, var_list=self.D_params)
	self.Domain_text_train_op = self.Domain_text_optim.minimize(self.D_text_loss)
	log_dir = os.path.join('.', 'logs', self.model_name)
	if not os.path.exists(log_dir):
            os.makedirs(log_dir)
	#### Old version
	self.writer = tf.train.SummaryWriter(os.path.join(log_dir, "SeqGAN_sample"), self.sess.graph)
        self.summary_op = tf.merge_all_summaries()
	tf.initialize_all_variables().run()
	if self.load_pretrain:
	    print "[@] Load the pretrained model"
	    self.G_saver = tf.train.Saver(self.G_params_dict)
	    self.G_saver.restore(self.sess, "./checkpoint/mscoco/G_pretrained/G_Pretrained-39000")

	self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
	count = 0
	D_count = 0
	G_count = 0
	for idx in range(self.max_iter//250):
            self.save(self.checkpoint_dir, count)
            self.evaluate(count)
            for _ in tqdm(range(250)):
		tgt_image_feature = self.dataset.flickr_sequential_sample(self.batch_size)
		tgt_text = self.dataset.flickr_caption_sequential_sample(self.batch_size)
		image_feature, right_text, _ = self.dataset.sequential_sample(self.batch_size)
                nonENDs = np.array(map(lambda x: (x != self.NOT).sum(), right_text))
                mask_t = np.zeros([self.batch_size, self.max_words])
                for ind, row in enumerate(mask_t):
		    # mask out the <BOS>
                    row[0:nonENDs[ind]] = 1

		wrong_text = self.dataset.get_wrong_text(self.batch_size)
		right_length = np.sum((right_text!=self.NOT)+0, 1)
                wrong_length = np.sum((wrong_text!=self.NOT)+0, 1)
		for _ in range(1):	# g_step
		    # update G
		    feed_dict = {self.images: tgt_image_feature}
		    _, G_loss = self.sess.run([self.G_train_op, self.G_loss], feed_dict)
		    G_count += 1
		for _ in range(20):      # d_step    
		    # update D
		    feed_dict = {self.images: image_feature, 
			self.right_text:right_text, 
			self.wrong_text:wrong_text, 
			self.right_length:right_length,
			self.wrong_length:wrong_length,
			self.mask: mask_t,
			self.src_images: image_feature,
			self.tgt_images: tgt_image_feature,
			self.src_text: right_text,
			self.tgt_text: tgt_text}

		    _, D_loss = self.sess.run([self.D_train_op, self.D_loss], feed_dict)
		    D_count += 1
		    _, D_text_loss = self.sess.run([self.Domain_text_train_op, self.D_text_loss], \
			    {self.src_text: right_text,
			    self.tgt_text: tgt_text,
			    self.images: tgt_image_feature
			    })

		count += 1

    def evaluate(self, count):
	
        samples = []
        samples_index = []
	image_feature, image_id, test_annotation = self.dataset.get_test_for_eval()
	num_samples = self.dataset.num_test_images
	samples_index = np.full([self.batch_size*(num_samples//self.batch_size), self.max_words], self.NOT)
        for i in range(num_samples//self.batch_size):
	    image_feature_test = image_feature[i*self.batch_size:(i+1)*self.batch_size]
	    feed_dict = {self.images: image_feature_test}
            predict_words = self.sess.run(self.predict_words_argmax, feed_dict)
            for j in range(self.batch_size):
		samples.append([self.dataset.decode(predict_words[j, :], type='string', remove_END=True)[0]])
                sample_index = self.dataset.decode(predict_words[j, :], type='index', remove_END=False)[0]
                samples_index[i*self.batch_size+j][:len(sample_index)] = sample_index
        # predict from samples
        samples = np.asarray(samples)
        samples_index = np.asarray(samples_index)
        print '[%] Sentence:', samples[0]
	meteor_pd = {}
        meteor_id = []
        for j in range(len(samples)):
            if image_id[j] == 0:
                break
            meteor_pd[str(int(image_id[j]))] = [{'image_id':str(int(image_id[j])), 'caption':samples[j][0]}]
            meteor_id.append(str(int(image_id[j])))
        scorer = COCOEvalCap(test_annotation, meteor_pd, meteor_id)
	scorer.evaluate(verbose=True)
        sample_dir = os.path.join("./SeqGAN_samples_sample", self.model_name)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        file_name = "%s_%s" % (self.dataset.dataset_name, str(count))
        np.savez(os.path.join(sample_dir, file_name), string=samples, index=samples_index, id=meteor_id)

    def save(self, checkpoint_dir, step):
        model_name = "SeqGAN_sample"
        model_dir = "%s" % (self.dataset.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir, self.model_name)
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
